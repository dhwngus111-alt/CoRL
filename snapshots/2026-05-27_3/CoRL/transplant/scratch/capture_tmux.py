import subprocess

def main():
    try:
        res = subprocess.run(["tmux", "capture-pane", "-t", "test", "-p", "-S", "-100"], capture_output=True, text=True)
        with open("transplant/scratch/tmux_output.txt", "w") as f:
            f.write(res.stdout)
            f.write("\n--- STDERR ---\n")
            f.write(res.stderr)
        print("Success")
    except Exception as e:
        with open("transplant/scratch/tmux_output.txt", "w") as f:
            f.write(str(e))
        print("Failed:", e)

if __name__ == "__main__":
    main()
