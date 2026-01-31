"""LLaMA Server GUI - Main entry point."""

import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='LLaMA Server GUI')
    parser.add_argument(
        '--ui',
        choices=['flextabs', 'legacy'],
        default='flextabs',
        help='UI implementation to use (default: flextabs)'
    )
    args = parser.parse_args()

    if args.ui == 'flextabs':
        try:
            from ui.flextabs_ui import FlextabsApp
            app = FlextabsApp()
            app.run()
        except ImportError as e:
            print(f"Error loading flextabs UI: {e}")
            print("Install dependencies: pip install flextabs ttkbootstrap")
            sys.exit(1)
    elif args.ui == 'legacy':
        print("Legacy UI not yet implemented")
        sys.exit(1)

if __name__ == '__main__':
    main()