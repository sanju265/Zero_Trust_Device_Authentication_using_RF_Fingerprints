"""
Admin Login Page for RF Fingerprint Zero Trust System.
Requires admin authentication before the monitoring dashboard starts.
"""

import os
import sys
import getpass
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich import box

from config import ADMIN_USERNAME, ADMIN_PASSWORD

console = Console()

LOGIN_BANNER = """
[bold cyan]
    ██████╗ ███████╗    ███████╗██╗███╗   ██╗ ██████╗ ███████╗██████╗ ██████╗ ██████╗ ██╗███╗   ██╗████████╗
    ██╔══██╗██╔════╝    ██╔════╝██║████╗  ██║██╔════╝ ██╔════╝██╔══██╗██╔══██╗██╔══██╗██║████╗  ██║╚══██╔══╝
    ██████╔╝█████╗      █████╗  ██║██╔██╗ ██║██║  ███╗█████╗  ██████╔╝██████╔╝██████╔╝██║██╔██╗ ██║   ██║
    ██╔══██╗██╔══╝      ██╔══╝  ██║██║╚██╗██║██║   ██║██╔══╝  ██╔══██╗██╔═══╝ ██╔══██╗██║██║╚██╗██║   ██║
    ██║  ██║██║         ██║     ██║██║ ╚████║╚██████╔╝███████╗██║  ██║██║     ██║  ██║██║██║ ╚████║   ██║
    ╚═╝  ╚═╝╚═╝         ╚═╝     ╚═╝╚═╝  ╚═══╝ ╚═════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝   ╚═╝
[/bold cyan]
"""

SYSTEM_TITLE = """[bold white]
    WiFi Device Impersonation Detection System
    Zero Trust Enterprise Security Framework
[/bold white]"""


def clear_screen():
    os.system("clear" if os.name == "posix" else "cls")


def show_login_screen():
    """Display the admin login page and authenticate the user."""
    max_attempts = 3

    for attempt in range(max_attempts):
        clear_screen()
        console.print(LOGIN_BANNER)
        console.print(SYSTEM_TITLE, justify="center")
        console.print()

        login_box = Panel(
            Align.center(Text.from_markup(
                "[dim]Authorized Personnel Only[/dim]\n"
                "[dim]Enterprise WiFi Security Monitoring Console[/dim]\n\n"
                f"[yellow]Attempt {attempt + 1} of {max_attempts}[/yellow]"
            )),
            title="[bold white] ADMIN LOGIN [/bold white]",
            border_style="cyan",
            box=box.DOUBLE,
            width=60,
            padding=(1, 2),
        )
        console.print(Align.center(login_box))
        console.print()

        # Get credentials
        try:
            console.print("  [bold cyan]Username:[/bold cyan] ", end="")
            username = input()
            console.print("  [bold cyan]Password:[/bold cyan] ", end="")
            password = getpass.getpass(prompt="")
        except (EOFError, KeyboardInterrupt):
            console.print("\n\n  [red]Login cancelled.[/red]")
            sys.exit(0)

        # Validate
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            console.print()
            success_panel = Panel(
                Align.center(Text.from_markup(
                    f"[bold green]ACCESS GRANTED[/bold green]\n\n"
                    f"Welcome, [bold]{username}[/bold]!\n"
                    f"Initializing security monitoring system..."
                )),
                border_style="green",
                box=box.DOUBLE,
                width=50,
                padding=(1, 2),
            )
            console.print(Align.center(success_panel))
            console.print()
            import time
            time.sleep(2)
            return True
        else:
            console.print()
            fail_panel = Panel(
                Align.center(Text.from_markup(
                    "[bold red]ACCESS DENIED[/bold red]\n"
                    "Invalid username or password."
                )),
                border_style="red",
                box=box.DOUBLE,
                width=50,
                padding=(1, 2),
            )
            console.print(Align.center(fail_panel))
            import time
            time.sleep(2)

    # Max attempts exceeded
    clear_screen()
    console.print()
    lockout_panel = Panel(
        Align.center(Text.from_markup(
            "[bold red]ACCOUNT LOCKED[/bold red]\n\n"
            "Maximum login attempts exceeded.\n"
            "Contact your system administrator."
        )),
        border_style="red",
        box=box.DOUBLE,
        width=50,
        padding=(1, 2),
    )
    console.print(Align.center(lockout_panel))
    console.print()
    sys.exit(1)
