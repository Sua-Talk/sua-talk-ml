from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
import datetime

def create_calendar_event(insights):
    # expects: insights['next_feeding'], insights['sleep_time'] as RFC3339 datetime string
    creds = Credentials.from_authorized_user_file('token.json', ['https://www.googleapis.com/auth/calendar'])
    service = build('calendar', 'v3', credentials=creds)
    if 'next_feeding' in insights:
        event = {
            'summary': 'Feeding Time',
            'start': {'dateTime': insights['next_feeding']},
            'end': {'dateTime': (datetime.datetime.fromisoformat(insights['next_feeding']) + datetime.timedelta(minutes=30)).isoformat()},
            'reminders': {'useDefault': True}
        }
        service.events().insert(calendarId='primary', body=event).execute()
