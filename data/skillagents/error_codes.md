# SkillAgents AI Error Codes

This page is the reference for every error code the SkillAgents AI platform can return. If the error persists after you follow the "What to do" column, email support@skillagents.ai with your order ID.

| Code | What it means | What to do |
| --- | --- | --- |
| E-1001 | Webhook delivery failed. Your integration endpoint did not return a 2xx response within fifteen seconds. | Confirm your endpoint is reachable and returns 2xx within the timeout. Retry the event from the Integrations page. |
| E-2301 | Cohort at capacity. You are on the waitlist. | No action required. When a seat opens, you are moved into the cohort and you receive an email. |
| E-3005 | Missing required field in enrollment form. A mandatory field was blank on submit. | Return to the enrollment page, complete every field marked with a red asterisk, and submit again. |
| E-4012 | Payment method declined by your bank. | Call your bank, ask them to approve SkillAgents AI as a merchant, and retry. If the retry fails, try a different card. |
| E-5001 | Video playback failed. Check your network. | Refresh the page. If playback still fails, test your link on fast.com. A stable five Mbps connection is enough. Corporate networks sometimes block the CDN, in which case try a personal network. |
| E-6002 | Rate limit exceeded on API. You sent more requests than your plan allows in a one minute window. | Back off and retry in sixty seconds. For a higher limit, email sales@skillagents.ai. |
| E-7829 | SSO token expired, please re-authenticate. | Sign out and sign back in through your identity provider. The new token is valid for eight hours. |

If an error you see is not listed here, it is likely a transient network issue. Wait thirty seconds and retry once before contacting support.
