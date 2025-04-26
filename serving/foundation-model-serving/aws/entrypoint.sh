#!/bin/bash
# Remount /dev/shm with a larger size (e.g., 1G)
echo "Remounting /dev/shm with 1G size"
mount -o remount,size=1G /dev/shm

# Execute the provided command
exec "$@"