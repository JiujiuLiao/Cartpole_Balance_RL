// DC Motor Controller for IBT-2 + Encoder
// Serial protocol (same format as stepper firmware):
//   <v,PWM>   set motor velocity, PWM = -255 to 255
//   <h,0>     reset encoder position to 0
//   <p,0>     return current encoder position
// All commands are echoed back after execution.

const int RPWM = 16;
const int LPWM = 17;
const int ENCODER_A = 18;
const int ENCODER_B = 19;

volatile long encoderPos = 0;

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

  Serial.begin(115200);
  pinMode(ENCODER_A, INPUT_PULLUP);
  pinMode(ENCODER_B, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(ENCODER_A), encoderISR, RISING);
  Serial.println("DC Motor Controller Ready");
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
    }

    returnParsedData();
    newData = false;
  }
}