// DC Motor Controller for IBT-2 + Encoder
// Serial protocol (same format as stepper firmware):
//   <v,PWM>   set motor velocity, PWM = -255 to 255
//   <h,0>     reset encoder position to 0
//   <p,0>     return current encoder position
//   <a,PWM>   acceleration burst test (ESP32-timed sampling, dumps CSV)
// All commands are echoed back after execution.
//
// Position streaming (added 2026-06-23):
//   The controller AUTO-PUSHES the encoder position as "P<pos>\n" at ~100 Hz,
//   without being asked. The Pi reads this stream in a background thread and
//   caches the latest value, so the control loop never blocks on a <p,0>
//   round-trip. The <p,0> query path is kept for backward compatibility.

const int RPWM = 16;
const int LPWM = 17;
const int ENCODER_A = 18;
const int ENCODER_B = 19;

volatile long encoderPos = 0;

// ── Acceleration burst test (command <a,PWM>) ─────
// ESP32-side self-timed sampling: micros() + own encoder into RAM, no serial
// in the timing loop, so Pi<->ESP32 latency does NOT affect the measurement.
const unsigned int  ACCEL_SAMPLE_US  = 1000;   // 1 ms sampling grid
const long          ACCEL_MAX_PULSES = 962;    // auto-brake at ~22 cm for current 80T setup
const unsigned long ACCEL_MAX_MS     = 700;    // time backstop
const int           ACCEL_MAX_SAMPLES = 800;   // 800*(4+4) = 6.4 KB
unsigned long accel_t[ACCEL_MAX_SAMPLES];      // microseconds since step
long          accel_pos[ACCEL_MAX_SAMPLES];    // encoder pulses

void IRAM_ATTR encoderISR() {
  if (digitalRead(ENCODER_B) == HIGH) {
    encoderPos++;
  } else {
    encoderPos--;
  }
}

// ── Serial parser ─────────────────────────────────
const byte MAX_CHARS = 32;
char receivedChars[MAX_CHARS];
char tempChars[MAX_CHARS];
bool newData = false;

char messageType[4];
long messageValue = 0;

void recvWithStartEndMarkers() {
  static bool recvInProgress = false;
  static byte ndx = 0;
  char startMarker = '<';
  char endMarker = '>';
  char rc;

  while (Serial.available() > 0 && !newData) {
    rc = Serial.read();
    if (recvInProgress) {
      if (rc != endMarker) {
        receivedChars[ndx++] = rc;
        if (ndx >= MAX_CHARS) ndx = MAX_CHARS - 1;
      } else {
        receivedChars[ndx] = '\0';
        ndx = 0;
        recvInProgress = false;
        newData = true;
      }
    } else if (rc == startMarker) {
      recvInProgress = true;
    }
  }
}

void parseData() {
  char *strtokIdx;
  strtokIdx = strtok(tempChars, ",");
  strncpy(messageType, strtokIdx, sizeof(messageType) - 1);
  strtokIdx = strtok(NULL, ",");
  messageValue = atol(strtokIdx);
}

// ── Motor control ─────────────────────────────────
void setMotor(int pwm) {
  pwm = constrain(pwm, -255, 255);
  if (pwm > 0) {
    analogWrite(RPWM, pwm);
    analogWrite(LPWM, 0);
  } else if (pwm < 0) {
    analogWrite(RPWM, 0);
    analogWrite(LPWM, -pwm);
  } else {
    analogWrite(RPWM, 0);
    analogWrite(LPWM, 0);
  }
}

// ── Command handlers ──────────────────────────────
void cmd_velocity() {
  setMotor((int)messageValue);
}

void cmd_home() {
  setMotor(0);
  encoderPos = 0;
}

void cmd_position() {
  Serial.print(encoderPos);
  Serial.print('\n');
}

// Acceleration burst: apply <a,PWM> step from rest, sample (micros, encoderPos)
// at a true 1 ms grid into RAM, auto-brake at distance/time limit, dump CSV.
void cmd_accel() {
  int pwm = constrain((int)messageValue, -255, 255);
  encoderPos = 0;
  int n = 0;
  unsigned long t0 = micros();
  unsigned long next = 0;
  setMotor(pwm);                                  // t=0 is stamped here, locally
  while (n < ACCEL_MAX_SAMPLES) {
    while ((micros() - t0) < next) { /* busy-wait to the 1 ms grid */ }
    unsigned long t = micros() - t0;
    long p = encoderPos;                          // volatile read
    accel_t[n]   = t;
    accel_pos[n] = p;
    n++;
    next += ACCEL_SAMPLE_US;
    if (labs(p) >= ACCEL_MAX_PULSES) break;        // distance auto-brake
    if (t >= ACCEL_MAX_MS * 1000UL) break;         // time backstop
  }
  setMotor(0);                                     // brake (both PWM low)

  Serial.print("ACCEL_BEGIN,pwm=");
  Serial.print(pwm);
  Serial.print(",n=");
  Serial.println(n);
  Serial.println("t_us,pos_pulses");
  for (int i = 0; i < n; i++) {
    Serial.print(accel_t[i]);
    Serial.print(',');
    Serial.println(accel_pos[i]);
  }
  Serial.println("ACCEL_END");
}

void returnParsedData() {
  // Echo back the command so Python can confirm receipt
  Serial.print('<');
  Serial.print(messageType);
  Serial.print(',');
  Serial.print(messageValue);
  Serial.print('>');
  Serial.print('\n');
}

// ── Setup & loop ──────────────────────────────────
void setup() {
  // Set motor pins LOW immediately — before anything else —
  // to prevent motor running during bootloader/init phase.
  pinMode(RPWM, OUTPUT);
  pinMode(LPWM, OUTPUT);
  analogWrite(RPWM, 0);
  analogWrite(LPWM, 0);

  // 2026-06-05: raise PWM from default (~1 kHz) to 20 kHz.
  // Motor L/R ≈ 5 ms — at 1 kHz current never reaches DC equiv,
  // giving only ~5 N effective force (1/26 of nameplate 139 N).
  // 20 kHz is above the motor electrical bandwidth and well below
  // the BTS7960 driver limit (~25 kHz).
  analogWriteFrequency(RPWM, 20000);
  analogWriteFrequency(LPWM, 20000);

  Serial.begin(115200);
  pinMode(ENCODER_A, INPUT_PULLUP);
  pinMode(ENCODER_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENCODER_A), encoderISR, RISING);
  Serial.println("DC Motor Controller Ready");
  Serial.println("PWM frequency: 20000 Hz (firmware build 2026-06-05)");
}

// Auto-push position so the Pi never has to block on a <p,0> round-trip.
// "P<pos>\n" — distinct from command echoes (which start with '<'), so the
// Pi's background reader can pick position lines out of the mixed stream.
const unsigned long POS_PUSH_INTERVAL_MS = 10;   // ~100 Hz
unsigned long lastPosPush = 0;

void pushPosition() {
  unsigned long nowMs = millis();
  if (nowMs - lastPosPush >= POS_PUSH_INTERVAL_MS) {
    lastPosPush = nowMs;
    Serial.print('P');
    Serial.print(encoderPos);   // 32-bit volatile long: read is atomic on ESP32
    Serial.print('\n');
  }
}

void loop() {
  recvWithStartEndMarkers();
  if (newData) {
    strcpy(tempChars, receivedChars);
    parseData();

    if (strcmp(messageType, "v") == 0) {
      cmd_velocity();
    } else if (strcmp(messageType, "h") == 0) {
      cmd_home();
    } else if (strcmp(messageType, "p") == 0) {
      cmd_position();
    } else if (strcmp(messageType, "a") == 0) {
      cmd_accel();
    }

    returnParsedData();
    newData = false;
  }

  pushPosition();
}
