import requests
import json

# Replace 'YOUR_WEBHOOK_URL' with the actual URL of your Discord webhook
DISCORD_WEBHOOK_URL = 'https://discord.com/api/webhooks/1108425291552010251/K5z6vd1Z9XG5HDgCWMyEB4_-H-E1-g3iikzC_rV7UmSnFqD-xcDZCmecOTjg_GcJctr0'

def send_embedded_message(description: str, statsMessage: str|dict, mentionTeam: bool = False):
  try:

    headers = {
        'Content-Type': 'application/json',
    }

    # Example usage with an embed
    if isinstance(statsMessage, str):

        embed = {
            'title': 'THESIS TRAINING NOTIFICATION',
            'description': description,
            'color': 1127128,
            'fields': [
                {'name': 'Stats', 'value': statsMessage, 'inline': True},
            ]
        }
    else:
        # value is a dictionary
        embed = {
            'title': 'THESIS TRAINING NOTIFICATION',
            'description': description,
            'color': 1127128,
            'fields': [
                {'name': key, 'value': str(value), 'inline': True} for key, value in statsMessage.items()
            ]
        }


    payload = {
        'embeds': [embed],
    }

    if mentionTeam:
        payload['content'] = '<@&1003310590753788075> Training Complete'

    
    response = requests.post(DISCORD_WEBHOOK_URL, headers=headers, data=json.dumps(payload))

    if response.status_code == 204:
        print("Embedded message sent successfully")
    else:
        print(f"Failed to send embedded message. Status code: {response.status_code}")
  except:
    pass
