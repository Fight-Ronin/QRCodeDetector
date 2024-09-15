import cv2
import numpy as np
from pyzbar.pyzbar import decode

def parse_vcard(data):
    # Simplified parsing, real vCard data may require more sophisticated parsing
    lines = data.split('\n')
    vcard_info = {}
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            vcard_info[key.strip()] = value.strip()
    return vcard_info

def parse_wifi(data):
    # Example for WiFi, real parsing may need to handle more variations
    components = data[5:].split(';')  # Remove "WIFI:" prefix
    wifi_info = {}
    for component in components:
        if component and ':' in component:
            key, value = component.split(':', 1)
            wifi_info[key.strip()] = value.strip()
    return wifi_info

def decode_and_categorize_qr_codes(bboxes, inp_img):
    decoded_objects = []
    for bbox in bboxes:
        # Extract the QR code image region using the bounding box
        x, y, w, h = cv2.boundingRect(bbox.astype(np.int32))
        qr_region = inp_img[y:y+h, x:x+w]

        # Decode QR codes from the region
        decoded = decode(qr_region)
        decoded_objects.extend(decoded)
    
    categorized_data = []
    for obj in decoded_objects:
        data = obj.data.decode("utf-8")
        if data.startswith("http://") or data.startswith("https://") or data.startswith("www://"):
            categorized_data.append({"type": "URL", "data": data})
        elif data.startswith("BEGIN:VCARD"):
            categorized_data.append({"type": "VCARD", "data": parse_vcard(data)})
        elif data.startswith("GEO:"):
            categorized_data.append({"type": "GEO", "data": data})
        elif data.startswith("WIFI:"):
            categorized_data.append({"type": "WIFI", "data": parse_wifi(data)})
        elif data.startswith("TEL:"):
            categorized_data.append({"type": "TEL", "data": data[4:]})
        elif data.startswith("SMSTO:"):
            sms_info = data.split(':', 2)
            if len(sms_info) == 3:
                categorized_data.append({"type": "SMS", "data": {"number": sms_info[1], "message": sms_info[2]}})
        elif data.startswith("MAILTO:"):
            email_info = data[7:].split('?')
            email_address = email_info[0]
            email_subject = email_info[1].split('=')[1] if len(email_info) > 1 else ""
            categorized_data.append({"type": "EMAIL", "data": {"address": email_address, "subject": email_subject}})
        else:
            categorized_data.append({"type": "TEXT", "data": data})
    
    return categorized_data