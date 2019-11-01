#include <iostream>
#include "rtmidi/RtMidi.h"
int main() {
  try {
    RtMidiIn midiin;
    std::cout << "good" << std::endl;
  } catch (RtMidiError &error) {
    // Handle the exception here
    error.printMessage();
  }
  return 0;
}