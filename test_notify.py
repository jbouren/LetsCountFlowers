#!/usr/bin/env python3
"""Test iMessage notification"""

import subprocess
import sys

def send_imessage(phone_number, message):
    script = f'''
    tell application "Messages"
        set targetService to 1st account whose service type = iMessage
        set targetBuddy to participant "{phone_number}" of targetService
        send "{message}" to targetBuddy
    end tell
    '''
    subprocess.run(["osascript", "-e", script])

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_notify.py '+1234567890'")
        sys.exit(1)
    
    phone = sys.argv[1]
    msg = "🌸 Test notification from Zinnia trainer!"
    
    print(f"Sending to {phone}: {msg}")
    send_imessage(phone, msg)
    print("Sent!")
