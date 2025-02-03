# Step-by-Step Guide to Create `credentials.json`

### 1. Go to the Google Cloud Console
- Open your browser and go to [Google Cloud Console](https://console.cloud.google.com/).
- If you don’t have a Google Cloud account, you may need to create one.

### 2. Create a New Project
- In the top-left corner, click on the project dropdown.
- Click on **"New Project"**.
- Name your project (e.g., "Job Interview Scheduler").
- Click **Create**.

### 3. Enable Gmail and Calendar APIs
- In the Google Cloud Console, go to the **API & Services** section (left sidebar).
- Click on **"Library"**.
- In the search bar, type:
  - **Gmail API** → Click on it and then click **Enable**.
  - **Google Calendar API** → Click on it and then click **Enable**.

### 4. Create OAuth 2.0 Credentials
- After enabling the APIs, go to **"Credentials"** in the **API & Services** section.
- Click on **"Create Credentials"** at the top and select **OAuth 2.0 Client IDs**.

### 5. Configure OAuth Consent Screen
- You will be prompted to configure the OAuth consent screen:
  - **User Type**: Choose **External** (for applications that will be used outside your organization).
  - **App Name**: Enter a name (e.g., "Job Interview Scheduler").
  - **User Support Email**: Enter your email address.
  - **Scopes**: For now, you can leave it as default (we’ll add scopes later in the code).
  - **Test Users**: You can add your email as a test user.
  - Click **Save and Continue**.

#### Create OAuth CredentialsDesktop Application OAuth Setup:

###### 1. **Create OAuth Credentials:**
- For **Application Type**, select **Desktop app** instead of **Web application**.
- You **do not need to add an Authorized Redirect URI** manually like you do for a web app.

###### 2. **Download the Credentials:**
- After setting the application type to **Desktop app**, click **Create**.
- You will see the **OAuth 2.0 Client ID** and the **Client Secret**.
- Click **Download** to save the `credentials.json` file to your computer.

###### 3. **Using the OAuth Credentials in the Code:**
- The `credentials.json` file will contain the necessary credentials to authenticate and interact with Google APIs (Gmail and Calendar in your case).

