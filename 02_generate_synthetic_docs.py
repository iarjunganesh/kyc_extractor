"""
Step 2: Generate synthetic passport documents for testing
Creates sample passport images without needing real documents
"""

from faker import Faker
from PIL import Image, ImageDraw, ImageFont
import os
from datetime import datetime, timedelta

fake = Faker('sv_SE')  # Swedish locale

def create_synthetic_passport(output_path="sample_passport.png"):
    """Create a synthetic passport image"""
    
    # Create image
    img = Image.new('RGB', (800, 500), color=(220, 220, 220))
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font, fallback to default if not available
    try:
        title_font = ImageFont.truetype("arial.ttf", 28)
        label_font = ImageFont.truetype("arial.ttf", 16)
        value_font = ImageFont.truetype("arial.ttf", 14)
    except:
        title_font = ImageFont.load_default()
        label_font = ImageFont.load_default()
        value_font = ImageFont.load_default()
    
    # Generate fake data
    name = fake.name()
    dob = fake.date_of_birth(minimum_age=18, maximum_age=80)
    passport_num = fake.passport_number()
    issue_date = fake.date_between(start_date='-10y', end_date='-5y')
    # Expiry date: between 1-5 years from now
    today = datetime.now().date()
    expiry_date = today + timedelta(days=fake.random_int(min=365, max=1825))
    
    # Draw title
    draw.text((50, 20), "PASSPORT", fill=(0, 0, 0), font=title_font)
    
    # Draw border
    draw.rectangle([(30, 10), (770, 490)], outline=(0, 0, 0), width=3)
    
    # Draw fields
    y_pos = 80
    line_height = 60
    
    fields = [
        ("Name:", name),
        ("Date of Birth:", str(dob)),
        ("Passport Number:", passport_num),
        ("Nationality:", "Sweden"),
        ("Issue Date:", str(issue_date)),
        ("Expiry Date:", str(expiry_date)),
    ]
    
    for label, value in fields:
        draw.text((60, y_pos), label, fill=(0, 0, 0), font=label_font)
        draw.text((250, y_pos), value, fill=(50, 50, 50), font=value_font)
        y_pos += line_height
    
    # Save image
    img.save(output_path)
    
    # Return the text data as well
    text_data = f"""PASSPORT
Name: {name}
Date of Birth: {dob}
Passport Number: {passport_num}
Nationality: Sweden
Issue Date: {issue_date}
Expiry Date: {expiry_date}"""
    
    return text_data, output_path

if __name__ == "__main__":
    print("Generating synthetic passport documents...")
    
    # Create documents folder if it doesn't exist
    docs_folder = "sample_documents"
    if not os.path.exists(docs_folder):
        os.makedirs(docs_folder)
    
    # Generate 3 sample passports
    for i in range(1, 4):
        output_path = os.path.join(docs_folder, f"passport_{i}.png")
        text_data, path = create_synthetic_passport(output_path)
        print(f"\n✓ Created: {path}")
        print(f"Content preview:\n{text_data[:100]}...")
    
    print(f"\n✓ All documents created in '{docs_folder}/' folder")
