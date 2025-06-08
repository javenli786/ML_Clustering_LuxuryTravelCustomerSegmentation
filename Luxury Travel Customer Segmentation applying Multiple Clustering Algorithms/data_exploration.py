import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def explore_data(df_guest_train, df_guest, df_booking):
    """
    Visualize the data
    """
    
    # df_guest_train
    
    # Distribution of GuestWithBooking
    Counts_GWB = df_guest_train['GuestWithBooking'].value_counts(normalize=True).sort_values(ascending=False).mul(100).nlargest(2)
    print("Distribution of GuestWithBooking (%): \n", Counts_GWB)

    # Distribution of Country
    country_percentage = df_guest_train['Country'].value_counts(normalize=True).sort_values(ascending=False).mul(100).nlargest(5).round(3)
    print("Distribution of Country (%): \n", country_percentage)
    plt.figure()
    country_percentage.plot(kind='bar')
    plt.title('Top 5 Country Percentage')
    plt.ylabel('Percentage')
    plt.show()

    # Distributio of BusinessUnitName
    BUN_percentage = df_guest_train['BusinessUnitName'].value_counts(normalize=True).sort_values(ascending=False).mul(100).round(3).nlargest(5)
    print("Distribution of BusinessUnitName (%): \n", BUN_percentage)
    plt.figure()
    BUN_percentage.plot(kind='bar')
    plt.title('Top 5 BusinessUnitName Percentage')
    plt.ylabel('Percentage')
    plt.show()

    # Distribution of SegmentTitle
    ST_percentage = df_guest_train['SegmentTitle'].value_counts(normalize=True).sort_values(ascending=False).mul(100).round(3).nlargest(6)
    print("Distribution of SegmentTitle (%): \n", ST_percentage)
    plt.figure()
    ST_percentage.plot(kind='bar')
    plt.title('Top 6 SegmentTitle Percentage')
    plt.ylabel('Percentage')
    plt.show()

    # Distribution of LoyaltyClub
    LC_percentage = df_guest_train['LoyaltyClub'].value_counts(normalize=True).sort_values(ascending=False).mul(100).round(3).nlargest(6)
    print("Distribution of LoyaltyClub (%): \n", LC_percentage)
    plt.figure()
    LC_percentage.plot(kind='bar')
    plt.title('Top 6 LoyaltyClub Percentage')
    plt.ylabel('Percentage')
    plt.show()

    # Distribution of RecordStatus
    RS_percentage = df_guest['RecordStatus'].value_counts(normalize=True).sort_values(ascending=False).mul(100).nlargest(8)
    print("Distribution of RecordStatus (%): \n", RS_percentage)
    plt.figure()
    RS_percentage.plot(kind='bar')
    plt.title('Top 8 RecordStatus Percentage')
    plt.ylabel('Percentage')
    plt.show()

    # 'df_booking'

    # Conversion rate of BookDate
    count_book_or_not = df_booking['BookDate'].isnull().value_counts()
    plt.figure(figsize=(6, 6))
    plt.pie(count_book_or_not, labels=['Not Converted', 'Converted'], autopct='%1.1f%%', startangle=140, colors=sns.color_palette('deep'), textprops={'fontsize': 6}, wedgeprops={'alpha':0.8})
    plt.title('Enquiry to Booking Conversion')
    plt.show()

    # Distribution of EnquiryDate
    plt.figure(figsize=(6, 6))
    sns.displot(df_booking['EnquiryDate'], kind='kde', color=sns.color_palette('deep')[0], fill=True)
    plt.grid(axis='y', linestyle='-', alpha=0.6)
    plt.xticks(rotation=90)
    plt.title('EnquiryDate Distribution')
    plt.show()

    # Distribution of MetaGroupName 
    MGN_percentage = df_booking['MetaGroupName'].value_counts(normalize=True).sort_values(ascending=False).mul(100).round(3).nlargest(10)
    print("Distribution of MetaGroupName (%): \n", MGN_percentage)
    plt.figure()
    MGN_percentage.plot(kind='bar')
    plt.title('Top 10 MetaGroupName Percentage')
    plt.ylabel('Percentage')
    plt.show()

    # Descriptive statistics of Adults, Child, Infant, Nights
    print("Descriptive statistics of Adults, Child, Infant, Nights \n", df_booking[['Adults', 'Child', 'Infant', 'Nights']].describe().round(3))

    # Boxenplot of Adults, Child, Infant, Nights
    df_people_long = pd.melt(df_booking, value_vars=['Adults', 'Child', 'Infant'], var_name='Category', value_name='Count')
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.boxenplot(data=df_people_long, x='Category', y='Count', palette='deep', saturation=0.8, ax=axes[0])
    sns.boxenplot(data=df_booking, y='Nights', palette='deep', saturation=0.8, ax=axes[1])
    axes[0].set_yscale('log')
    axes[1].set_yscale('log')
    axes[0].set_title('People Count')
    axes[1].set_title('Nights')
    plt.tight_layout()
    plt.show()