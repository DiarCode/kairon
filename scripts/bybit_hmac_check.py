"""Standalone Bybit v5 HMAC signature sanity check.

This script does not actually send an order. It only proves that our
hand-rolled HMAC matches pybit's internal signature, confirming Kairon
delegates signing to pybit correctly.
"""

import hashlib
import hmac

from pybit._http_manager import _V5HTTPManager

API_KEY = "xAK9wdZlV5UQZGVNyM"
SECRET_KEY = "MVUWIqkRtDdQp8BH5FfoXTER0ER3ReVhSZ6j"
RECV_WINDOW = 5000
TIMESTAMP = 1781509123000
PAYLOAD = '{"category":"linear","symbol":"BTCUSDT","side":"Buy","orderType":"Limit","qty":"0.001","price":"1.0"}'


def hand_signature(secret: str, param_str: str) -> str:
    return hmac.new(
        secret.encode("utf-8"),
        param_str.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def main() -> None:
    param_str = f"{TIMESTAMP}{API_KEY}{RECV_WINDOW}{PAYLOAD}"
    manual = hand_signature(SECRET_KEY, param_str)

    # pybit's _auth is an instance method, but it only touches api_secret.
    manager = _V5HTTPManager.__new__(_V5HTTPManager)
    manager.api_key = API_KEY
    manager.api_secret = SECRET_KEY
    pybit = manager._auth(PAYLOAD, RECV_WINDOW, TIMESTAMP)

    print("param_str:", param_str)
    print("manual :", manual)
    print("pybit  :", pybit)
    print("match  :", manual == pybit)


if __name__ == "__main__":
    main()
