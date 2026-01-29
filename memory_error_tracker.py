import logging
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from crewai.events import (
    BaseEventListener,
    MemorySaveFailedEvent,
    MemoryQueryFailedEvent
)

logger = logging.getLogger('memory_errors')

class MemoryErrorTracker(BaseEventListener):
    def __init__(self, notify_email: Optional[str] = None):
        super().__init__()
        self.notify_email = notify_email
        self.error_count = 0

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(MemorySaveFailedEvent)
        def on_memory_save_failed(source, event: MemorySaveFailedEvent):
            self.error_count += 1
            agent_info = f"Agent '{event.agent_role}'" if event.agent_role else "Unknown agent"
            error_message = f"Memory save failed: {event.error}. {agent_info}"
            logger.error(error_message)

            if self.notify_email and self.error_count % 5 == 0:
                self._send_notification(error_message)

        @crewai_event_bus.on(MemoryQueryFailedEvent)
        def on_memory_query_failed(source, event: MemoryQueryFailedEvent):
            self.error_count += 1
            error_message = f"Memory query failed: {event.error}. Query: '{event.query}'"
            logger.error(error_message)

            if self.notify_email and self.error_count % 5 == 0:
                self._send_notification(error_message)

    def _send_notification(self, message):
        if not self.notify_email:
            return

        try:
            smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
            smtp_port = int(os.getenv('SMTP_PORT', '587'))
            smtp_username = os.getenv('SMTP_USERNAME')
            smtp_password = os.getenv('SMTP_PASSWORD')
            sender_email = os.getenv('SMTP_FROM_EMAIL', smtp_username)

            if not smtp_username or not smtp_password:
                logger.warning("SMTP credentials not configured. Email notification skipped.")
                print(f"[NOTIFICATION] Email not sent - configure SMTP_USERNAME and SMTP_PASSWORD")
                return

            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = self.notify_email
            msg['Subject'] = f'CrewAI Memory Error Alert - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'

            body = f"""
Memory Error Alert

{message}

Total errors so far: {self.error_count}

This is an automated notification from your CrewAI application.
Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

            msg.attach(MIMEText(body, 'plain'))

            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.send_message(msg)

            logger.info(f"Notification email sent to {self.notify_email}")
            print(f"[NOTIFICATION] Email sent to {self.notify_email}")

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            print(f"[NOTIFICATION] Failed to send email: {e}")

# Create an instance of your listener
error_tracker = MemoryErrorTracker(notify_email="danilocastro81@gmail.com")