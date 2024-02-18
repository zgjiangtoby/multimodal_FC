import pandas as pd
import os

# Assuming path and df are already defined as in your example
path = "../mocheg/test"
df = pd.read_csv(path + "/Corpus2.csv")
#
images_folder_path = path + '/images/'  # Update this as needed
image_files = os.listdir(images_folder_path)

# Create a list of dictionaries for each image file
image_data = []
for img in image_files:
    claim_id = img.split("-")[0]  # Extract claim_id from the file name
    image_data.append({'claim_id': claim_id, 'image_file': img})

# Convert this list into a DataFrame
images_df = pd.DataFrame(image_data)

# Ensure claim_id types match between df and images_df
df['claim_id'] = df['claim_id'].astype(str)
images_df['claim_id'] = images_df['claim_id'].astype(str)

# Merge the original DataFrame with the images DataFrame on claim_id
# This will create a row for each claim_id-image pair
merged_df = pd.merge(df, images_df, on='claim_id', how='left')
merged_df.dropna(subset=['image_file'], inplace=True)
merged_df.dropna(subset=['Evidence'], inplace=True)
outpath = path + "/matched_Corpus2.csv"
merged_df.to_csv(outpath, index=False)