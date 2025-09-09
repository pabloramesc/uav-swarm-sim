import sys, signal


def exit_handler(sig, frame):
    sig_name = signal.Signals(sig).name
    print(f"\n🛑 Received signal: {sig_name} ({sig}) — exiting cleanly...")
    sys.exit(0)

def register_exit_signal(sig=signal.SIGINT):
    signal.signal(sig, exit_handler)
