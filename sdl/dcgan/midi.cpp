
#include "RtMidi.h"

// port = (unsigned int) atoi( argv[1] );
unsigned int port = 1; // change this to midi device port - see midiprobe executable to check it


if ( port >= nPorts ) {
    delete midiin;
    std::cout << "Invalid port specifier!\n";
    usage();
}

try {
    midiin->openPort( port );
}
catch ( RtMidiError &error ) {
    error.printMessage();
    goto cleanup;
}

