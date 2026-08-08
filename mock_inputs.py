import sys
import os

# Put mock inputs in a file to feed it
with open("mock_input.txt", "w") as f:
    f.write("2\n") # Load Dijkstra
    f.write("a\n") # Pick from downloaded
    f.write("qwen0.5b\n") # Select model
    f.write("y\n") # Train Router yes
    f.write("y\n") # Train Bridge yes
    f.write("5\n") # Exit

print("Inputs written.")
