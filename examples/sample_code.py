"""A deliberately flawed module — the demo input for the AI Code Review Bot.

Each function below hides a common, real-world problem (security, correctness, or
style) so the reviewer has concrete things to find. Do NOT use this code for real.
"""

import json  # unused import


def get_user(db, user_id):
    # SQL injection: user_id is interpolated straight into the query string.
    query = "SELECT * FROM users WHERE id = '%s'" % user_id
    return db.execute(query)


API_KEY = "sk-live-1234567890abcdef"  # hardcoded secret committed to source


def average(numbers=[]):  # mutable default argument (shared across calls)
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)  # ZeroDivisionError when the list is empty


def read_config(path):
    try:
        return json.load(open(path))  # file is never closed
    except:  # bare except hides real errors (e.g. KeyboardInterrupt)
        return None
