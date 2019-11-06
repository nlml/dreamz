import rtmidi


def get_rtmidi_obj(port):
    rtmidi_obj = rtmidi.RtMidiIn()
    rtmidi_obj.openPort(port)
    return rtmidi_obj


def get_midi_msg_stack(rtmidi_obj):
    midi_msg_stack = []
    count = 0
    while True:
        msg = rtmidi_obj.getMessage()
        if msg is None:
            # if count > 0:
            #     print(count)
            break
        else:
            midi_msg_stack.append(msg)
            count += 1
    return midi_msg_stack


def process_midi_event(event):
    if event.isNoteOn():
        return event.getNoteNumber()
    elif event.isController():
        return event.getControllerNumber(), event.getControllerValue()
    return None


def process_midi_msg_stack(rtmidi_obj):
    midi_msg_stack = get_midi_msg_stack(rtmidi_obj)
    out = []
    while len(midi_msg_stack):
        n = midi_msg_stack[0]
        event = process_midi_event(n)
        if event is not None:  # useful event
            out.append(event)
        midi_msg_stack = midi_msg_stack[1:]
    return out
