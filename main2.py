# example.py
def say_hello():
    print("Hello from module!")

print("This runs when the module is loaded.")

if __name__ == "__main__":
    print("This runs only when the script is run directly.")
    say_hello()