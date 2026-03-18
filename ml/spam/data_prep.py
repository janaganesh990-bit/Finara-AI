import pandas as pd
import numpy as np
import os

# Paths
SMS_SPAM_PATH = os.path.join('data', 'spam.tsv')
PHISHING_EMAIL_PATH = os.path.join('data', 'phishing_email.csv')
MERGED_DATA_PATH = os.path.join('data', 'scam_dataset.csv')

def prepare_dataset():
    # 1. Load SMS Spam dataset
    print("Loading SMS Spam dataset...")
    df_sms = pd.read_csv(SMS_SPAM_PATH, sep='\t', names=['label', 'text'])
    df_sms['scam_type'] = df_sms['label'].apply(lambda x: 'Legit' if x == 'ham' else 'General Spam')
    df_sms['label'] = df_sms['label'].map({'ham': 0, 'spam': 1})

    # 2. Load User-Provided Samples
    scam_samples = [
        ("Your KYC will expire today. Update immediately to avoid account suspension.", 1, "Fake KYC", "Fear"),
        ("Your SBI account will be blocked within 2 hours. Verify PAN now.", 1, "Fake KYC", "Urgency"),
        ("Click here to receive pre-approved loan of ₹25,000 instantly.", 1, "Loan Scam", "Urgency"),
        ("Congratulations! You won ₹10,00,000 in lucky draw.", 1, "Lottery Scam", "Greed"),
        ("Double your money in 15 days. Limited seats available.", 1, "Ponzi", "Greed"),
        ("Send OTP to confirm your cashback reward.", 1, "UPI Collect Scam", "Urgency"),
        ("You have pending electricity bill. Pay now to avoid disconnection.", 1, "Bill Scam", "Fear"),
        ("Immediate action required: Income tax notice issued.", 1, "Tax Scam", "Fear"),
        ("Join our crypto group and earn 5% daily guaranteed.", 1, "Investment Scam", "Greed"),
        ("Work from home and earn ₹50,000 per week.", 1, "Job Scam", "Greed"),
        ("Your parcel is stuck. Pay ₹99 delivery fee now.", 1, "Delivery Scam", "Urgency"),
        ("Verify Aadhaar to continue using bank services.", 1, "Fake KYC", "Authority"),
        ("Limited time offer! Invest ₹5,000 and get ₹50,000 return.", 1, "Ponzi", "Greed"),
        ("Your UPI collect request of ₹4,999 is pending. Approve now.", 1, "UPI Collect Scam", "Urgency"),
        ("You are selected for government subsidy. Pay processing fee.", 1, "Subsidy Scam", "Greed"),
        ("Account hacked! Reset password immediately.", 1, "Phishing", "Fear"),
        ("Your credit card reward points expire today. Redeem now.", 1, "Reward Scam", "Urgency"),
        ("Confirm your Netflix payment via this link.", 1, "Subscription Scam", "Trust"),
        ("Urgent: Bank server update. Share debit card details.", 1, "Phishing", "Authority"),
        ("Invest in gold bond scheme. 3x return guaranteed.", 1, "Investment Scam", "Greed")
    ]
    
    legit_samples = [
        ("Your salary of ₹75,000 has been credited.", 0, "Legit", "None"),
        ("Your Amazon order has been delivered successfully.", 0, "Legit", "None"),
        ("Netflix subscription of ₹499 processed successfully.", 0, "Legit", "None"),
        ("Your UPI payment of ₹850 to Zomato was successful.", 0, "Legit", "None"),
        ("Bank statement for November is now available.", 0, "Legit", "None"),
        ("EMI of ₹3,250 deducted successfully.", 0, "Legit", "None"),
        ("Your OTP for transaction is 482931.", 0, "Legit", "None"),
        ("Electricity bill of ₹1,250 generated.", 0, "Legit", "None"),
        ("Income tax refund of ₹2,340 processed.", 0, "Legit", "None"),
        ("Your SIP investment of ₹5,000 completed.", 0, "Legit", "None"),
        ("UPI payment to Swiggy completed.", 0, "Legit", "None"),
        ("Credit card bill of ₹8,950 due on 15th.", 0, "Legit", "None"),
        ("Your account balance is ₹1,25,000.", 0, "Legit", "None"),
        ("Loan EMI payment received.", 0, "Legit", "None"),
        ("ATM withdrawal of ₹2,000 successful.", 0, "Legit", "None"),
        ("FD interest credited.", 0, "Legit", "None"),
        ("Transaction alert: ₹1,200 spent at Amazon.", 0, "Legit", "None"),
        ("Payment received from John via UPI.", 0, "Legit", "None"),
        ("Recharge of ₹299 completed.", 0, "Legit", "None"),
        ("Your insurance premium has been paid.", 0, "Legit", "None")
    ]
    
    df_user = pd.DataFrame(scam_samples + legit_samples, columns=['text', 'label', 'scam_type', 'trigger'])
    
    # 3. Load Phishing Email dataset if available
    df_phish = pd.DataFrame()
    if os.path.exists(PHISHING_EMAIL_PATH):
        try:
            print("Loading Phishing Email dataset...")
            # Loading and cleaning varies by source, let's assume 'text' and 'label' columns
            # This is a placeholder for the actual format of the downloaded file
            df_p = pd.read_csv(PHISHING_EMAIL_PATH)
            if 'Email Text' in df_p.columns and 'Email Type' in df_p.columns:
                df_phish = df_p[['Email Text', 'Email Type']].rename(columns={'Email Text': 'text', 'Email Type': 'label'})
                df_phish['label'] = df_phish['label'].map({'Phishing Email': 1, 'Safe Email': 0})
                df_phish['scam_type'] = df_phish['label'].apply(lambda x: 'Phishing' if x == 1 else 'Legit')
        except Exception as e:
            print(f"Error loading phishing dataset: {e}")

    # 4. Merge all
    print("Merging datasets...")
    # Adjust SMS dataframe to match user-provided schema
    df_sms_clean = df_sms[['text', 'label', 'scam_type']].copy()
    df_sms_clean['trigger'] = 'None'
    
    # Ensure all have the same columns
    combined_df = pd.concat([df_sms_clean, df_user], ignore_index=True)
    if not df_phish.empty:
        df_phish['trigger'] = 'None'
        combined_df = pd.concat([combined_df, df_phish], ignore_index=True)

    # 5. Handle overlaps and duplicates
    combined_df = combined_df.drop_duplicates(subset=['text']).reset_index(drop=True)
    
    # 6. Final Clean and Save
    combined_df.to_csv(MERGED_DATA_PATH, index=False)
    print(f"Dataset prepared and saved to {MERGED_DATA_PATH}. Total records: {len(combined_df)}")

if __name__ == "__main__":
    prepare_dataset()
