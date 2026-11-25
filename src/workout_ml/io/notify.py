"""Email alerts."""
from __future__ import annotations

import logging
import smtplib
import ssl
from email.message import EmailMessage
from typing import Protocol

from ..config import AlertSettings

log = logging.getLogger(__name__)


class Notifier(Protocol):
    def send(self, subject: str, body: str) -> None: ...


class LogNotifier:
    """Used when alerts are disabled: the message still lands in the logs."""

    def send(self, subject, body):
        log.warning('alert (email disabled): %s', subject, extra={'alert_body': body})


class SmtpNotifier:
    def __init__(self, settings: AlertSettings):
        self._s = settings

    def send(self, subject, body):
        msg = EmailMessage()
        msg['Subject'], msg['From'], msg['To'] = subject, self._s.smtp_user, self._s.to
        msg.set_content(body)
        with smtplib.SMTP_SSL(self._s.smtp_host, self._s.smtp_port, timeout=30,
                              context=ssl.create_default_context()) as server:
            server.login(self._s.smtp_user, self._s.smtp_password.get_secret_value())
            server.send_message(msg)


def build_notifier(settings: AlertSettings) -> Notifier:
    return SmtpNotifier(settings) if settings.enabled else LogNotifier()
