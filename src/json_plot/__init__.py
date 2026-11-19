import subprocess
import sys


def main() -> None:
    print("What would you like to do?")
    print("1. Create data visualizations")
    print("2. Render annotated PDF pages")

    choice = input("Choose 1 or 2: ").strip()
    modules = {
        "1": "json_plot.visualize_data",
        "2": "json_plot.render_pages",
    }
    module = modules.get(choice)

    if module is None:
        print("Invalid choice. Enter 1 or 2.")
        return

    subprocess.run([sys.executable, "-m", module], check=True)
