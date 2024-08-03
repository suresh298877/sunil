from flask import Flask, render_template, request
import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
chat_history_ids = None  # Define chat history globally

@app.route("/")
def index():
    global chat_history_ids
    chat_history_ids = None  # Reset chat history
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    global chat_history_ids  # Access the global chat history
    user_input = request.form["msg"]

    # Generate response
    response = get_chat_response(user_input)

    return response

def get_chat_response(user_input):
    global chat_history_ids
    # Encode the user input
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    
    # Concatenate the user input with chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids
    
    # Generate response
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    
    # Decode and return response
    bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Conversation flow for FarMart
    if "hello" in user_input.lower() or "hi" in user_input.lower() or "hai" in user_input.lower() or "hii" in user_input.lower():
        bot_response = "Hi! Welcome to FarMart. How can I assist you today?"
    elif "upload" in user_input.lower():
        bot_response = "To upload your produce, please log in to your account and navigate to the 'Upload Produce' section."
    elif "price" in user_input.lower():
        bot_response = "Farmers can set their own prices for their produce directly on the FarMart platform."
    elif "search" in user_input.lower():
        bot_response = "You can search for fresh produce by using the search bar on our homepage."
    elif "location" in user_input.lower() or "map" in user_input.lower():
        bot_response = "Our site includes a map feature that shows where each farm is located. You can plan visits to pick up your purchases."
    elif "ai chatbot" in user_input.lower():
        bot_response = "Our AI-powered chatbot assists farmers in regional languages, making it convenient for everyone to use the platform."
    elif "support" in user_input.lower():
        bot_response = "For any support, please contact us at support@farmart.com."
    elif "objective" in user_input.lower():
        bot_response = """FarMart aims to:
1. Help farmers sell their produce directly to consumers, increasing their earnings.
2. Provide consumers with a simple way to find and buy fresh, locally grown produce.
3. Strengthen community connections and support local farming."""
    elif "features" in user_input.lower():
        bot_response = """FarMart features include:
1. Search and Filtering
2. Produce Listings
3. Farm Location Maps
4. Direct Communication
5. AI Chatbot
6. Reviews and Ratings
7. Community Engagement
8. Social Media Integration
9. Admin Dashboard
10. Farmer Profiles
11. Resource Center"""
    elif "methodology" in user_input.lower():
        bot_response = """To build FarMart, we:
1. Determine the needs of farmers and buyers.
2. Design and create a user-friendly interface.
3. Develop the website using HTML, CSS, and JavaScript.
4. Set up SQL to manage data.
5. Add key features like search tools, farm location maps, and an AI chatbot.
6. Test the site for functionality.
7. Deploy on cloud services like AWS or Google Cloud.
8. Provide support and training.
9. Continuously collect feedback and make improvements."""
    elif "date" in user_input.lower():
        bot_response = f"Today's date is {datetime.date.today()}."
    elif "bye" in user_input.lower() or "goodbye" in user_input.lower() or "see you" in user_input.lower():
        bot_response = "Goodbye! Have a great day. If you need further assistance, feel free to come back anytime!"
    elif "thank you" in user_input.lower() or "thanks" in user_input.lower():
        bot_response = "You're welcome! If you have any more questions, just let me know."
    elif "hindi" in user_input.lower() or "हिंदी" in user_input:
        bot_response = """FarMart वेबसाइट का उपयोग कैसे करें:
1. **वेबसाइट पर लॉग इन करें**: अपने खाते में लॉग इन करें या नया खाता बनाएँ।
2. **उत्पाद अपलोड करें**: 'अपलोड उत्पाद' अनुभाग पर जाएँ और अपने उत्पाद की जानकारी भरें।
3. **मूल्य सेट करें**: आप अपने उत्पाद के लिए मूल्य निर्धारित कर सकते हैं।
4. **खोजें और ब्राउज़ करें**: ताजे उत्पाद खोजने के लिए होमपेज पर सर्च बार का उपयोग करें।
5. **मानचित्र पर स्थान देखें**: प्रत्येक फार्म का स्थान मानचित्र पर देखें और योजनाबद्ध तरीके से खरीदारी करें।
6. **सहायता प्राप्त करें**: किसी भी सहायता के लिए हमें support@farmart.com पर संपर्क करें।"""
    
    elif "telugu" in user_input.lower() or "తెలుగు" in user_input:
        bot_response = """FarMart వెబ్సైట్‌ను ఎలా ఉపయోగించాలి:
1. **వెబ్సైట్‌లో లాగిన్ అవ్వండి**: మీ ఖాతాలో లాగిన్ అవ్వండి లేదా కొత్త ఖాతా సృష్టించండి.
2. **ఉత్పత్తులు అప్‌లోడ్ చేయండి**: 'అప్‌లోడ్ ఉత్పత్తులు' విభాగానికి వెళ్లండి మరియు మీ ఉత్పత్తి సమాచారం నింపండి.
3. **ధరను సెట్ చేయండి**: మీ ఉత్పత్తికి ధరను సెట్ చేయవచ్చు.
4. **శోధించండి మరియు బ్రౌజ్ చేయండి**: తాజా ఉత్పత్తులను కనుగొనడానికి హోమ్‌పేజ్‌పై సెర్చ్ బార్‌ను ఉపయోగించండి.
5. **మ్యాప్‌పై స్థానం చూడండి**: ప్రతి ఫార్మ్ యొక్క స్థానం మ్యాప్‌పై చూడండి మరియు మీ కొనుగోళ్లను ప్రణాళిక ప్రకారం నిర్వహించండి.
6. **సహాయం పొందండి**: ఏదైనా సహాయం కోసం మాకు support@farmart.com కు సంప్రదించండి."""

    elif "tamil" in user_input.lower() or "தமிழ்" in user_input:
        bot_response = """FarMart இணையதளத்தை எப்படி பயன்படுத்துவது:
1. **வெப்சைட் உள்நுழைவு**: உங்கள் கணக்கில் உள்நுழையுங்கள் அல்லது புதிய கணக்கு ஒன்றை உருவாக்குங்கள்.
2. **உற்பத்திகளை பதிவேற்றவும்**: 'பதிவேற்ற உற்பத்திகள்' பிரிவிற்குச் சென்று உங்கள் உற்பத்தி விவரங்களை நிரப்புங்கள்.
3. **விலை அமைக்கவும்**: உங்கள் உற்பத்திக்கு விலை நிர்ணயிக்கலாம்.
4. **தேடவும் மற்றும் உலாவவும்**: புதிய உற்பத்திகளை தேடுவதற்கு முகப்புப்பக்கம் உள்ள தேடல் தட்டையைப் பயன்படுத்துங்கள்.
5. **நகலில் இடத்தைப் பாருங்கள்**: ஒவ்வொரு பண்ணையின் இடத்தை மேப்பில் பார்க்கவும் மற்றும் உங்கள் வாங்குவதை திட்டமிடுங்கள்.
6. **உதவி பெறவும்**: எந்தவொரு உதவிக்கும் support@farmart.com க்கு அணுகவும்."""

    elif "kerala" in user_input.lower() or "മലയാളം" in user_input:
        bot_response = """FarMart വെബ്സൈറ്റ് എങ്ങനെ ഉപയോഗിക്കാം:
1. **വെബ്സൈറ്റിൽ ലോഗിൻ ചെയ്യുക**: നിങ്ങളുടെ അക്കൗണ്ടിൽ ലോഗിൻ ചെയ്യുക അല്ലെങ്കിൽ പുതിയ അക്കൗണ്ട് ഉണ്ടാക്കുക.
2. **ഉൽപ്പന്നങ്ങൾ അപ്‌ലോഡ് ചെയ്യുക**: 'അപ്‌ലോഡ് ഉൽപ്പന്നങ്ങൾ' വിഭാഗത്തിലേക്ക് പോകുക ಮತ್ತು നിങ്ങളുടെ ഉൽപ്പന്നത്തിന്റെ വിവരങ്ങൾ പൂരിപ്പിക്കുക.
3. **വില സെറ്റ് ചെയ്യുക**: നിങ്ങളുടെ ഉൽപ്പന്നത്തിന് വില നിശ്ചയിക്കാം.
4. **തിരയുക ಮತ್ತು ബ്രൗസ് ചെയ്യുക**: പുതിയ ഉൽപ്പന്നങ്ങൾ കണ്ടെത്താൻ ഹോംപേജ്上的 സെർച്ച് ബാർ ഉപയോഗിക്കുക.
5. **മാപ്പിൽ സ്ഥാനം കാണുക**: ഓരോ ഫാമിന്റെയും സ്ഥാനം മാപ്പിൽ കാണുക, നിങ്ങളുടെ വാങ്ങലുകൾ പ്ലാൻ ചെയ്യാൻ സഹായിക്കും.
6. **സഹായം നേടുക**: ഏത് സഹായത്തിനും support@farmart.com എന്ന വിലാസത്തിൽ ബന്ധപ്പെടുക."""

    elif "kannada" in user_input.lower() or "ಕನ್ನಡ" in user_input:
        bot_response = """FarMart ವೆಬ್ಸೈಟ್ ಅನ್ನು ಹೇಗೆ ಬಳಸುವುದು:
1. **ವೆಬ್ಸೈಟ್‌ನಲ್ಲಿ ಲಾಗಿನ್ ಮಾಡಿ**: ನಿಮ್ಮ ಖಾತೆಯಲ್ಲಿ ಲಾಗಿನ್ ಆಗಿ ಅಥವಾ ಹೊಸ ಖಾತೆ ತೆರೆಯಿರಿ.
2. **ಉತ್ಪತ್ತಿಗಳನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ**: 'ಅಪ್ಲೋಡ್ ಉತ್ಪತ್ತಿಗಳು' ವಿಭಾಗಕ್ಕೆ ಹೋಗಿ ಮತ್ತು ನಿಮ್ಮ ಉತ್ಪತ್ತಿಯ ವಿವರಗಳನ್ನು ಭರ್ತಿ ಮಾಡಿ.
3. **ದರವನ್ನು ಹೊಂದಿಸಿ**: ನಿಮ್ಮ ಉತ್ಪತ್ತಿಗೆ ದರವನ್ನು ಹೊಂದಿಸಬಹುದು.
4. **ಹುಡುಕು ಮತ್ತು ಬ್ರೌಸ್ ಮಾಡಿ**: ಹೊಸ ಉತ್ಪತ್ತಿಗಳನ್ನು ಹುಡುಕಲು ಹೋಮ್‌ಪೇಜ್‌ನಲ್ಲಿ ಹುಡುಕುವ ಬಾರ್ ಬಳಸಿಕೊಳ್ಳಿ.
5. **ಮಾಪ್‌ನಲ್ಲಿ ಸ್ಥಳವನ್ನು ನೋಡಿ**: ಪ್ರತಿ ಫಾರ್ಮ್‌ನ ಸ್ಥಳವನ್ನು ನಕ್ಷೆಯಲ್ಲಿ ನೋಡಿ ಮತ್ತು ನಿಮ್ಮ ಖರೀದಿಗಳನ್ನು ಯೋಜಿಸಲು ಸಹಾಯ ಮಾಡಿ.
6. **ಸಹಾಯ ಪಡೆಯಿರಿ**: ಯಾವುದೇ ಸಹಾಯಕ್ಕಾಗಿ support@farmart.com ಗೆ ಸಂಪರ್ಕಿಸಿ."""

    elif "bengali" in user_input.lower() or "বাংলা" in user_input:
        bot_response = """FarMart ওয়েবসাইট কীভাবে ব্যবহার করবেন:
1. **ওয়েবসাইটে লগইন করুন**: আপনার অ্যাকাউন্টে লগইন করুন অথবা একটি নতুন অ্যাকাউন্ট তৈরি করুন।
2. **পণ্য আপলোড করুন**: 'আপলোড পণ্য' বিভাগে যান এবং আপনার পণ্যের তথ্য পূরণ করুন।
3. **মূল্য নির্ধারণ করুন**: আপনার পণ্যের জন্য মূল্য নির্ধারণ করতে পারেন।
4. **অনুসন্ধান করুন এবং ব্রাউজ করুন**: নতুন পণ্য খুঁজতে হোমপেজে অনুসন্ধান বারের ব্যবহার করুন।
5. **মানচিত্রে অবস্থান দেখুন**: প্রতিটি ফার্মের অবস্থান মানচিত্রে দেখুন এবং আপনার কেনাকাটা পরিকল্পনা করুন।
6. **সাহায্য পান**: যেকোনো সাহায্যের জন্য support@farmart.com-এ যোগাযোগ করুন।"""

    elif "odia" in user_input.lower() or "ଓଡ଼ିଆ" in user_input:
        bot_response = """FarMart ୱେବସାଇଟ୍ କିପରି ବ୍ୟବହାର କରିବେ:
1. **ୱେବସାଇଟ୍ ରେ ଲଗ୍ ଇନ୍ କରନ୍ତୁ**: ଆପଣଙ୍କର ଖାତାରେ ଲଗ୍ ଇନ୍ କରନ୍ତୁ କିମ୍ବା ଏକ ନୂତନ ଖାତା ସୃଷ୍ଟି କରନ୍ତୁ।
2. **ଉତ୍ପାଦ ଅପ୍ଲୋଡ୍ କରନ୍ତୁ**: 'ଅପ୍ଲୋଡ୍ ଉତ୍ପାଦ' ଅଞ୍ଚଳକୁ ଯାଆନ୍ତୁ ଏବଂ ଆପଣଙ୍କର ଉତ୍ପାଦର ବିବରଣୀ ଭରନ୍ତୁ।
3. **ମୂଲ୍ୟ ନିର୍ଧାରଣ କରନ୍ତୁ**: ଆପଣଙ୍କର ଉତ୍ପାଦ ପାଇଁ ମୂଲ୍ୟ ନିର୍ଧାରଣ କରିପାରିବେ।
4. **ସନ୍ଧାନ କରନ୍ତୁ ଓ ବ୍ରାଉଜ୍ କରନ୍ତୁ**: ନୂତନ ଉତ୍ପାଦ ଖୋଜିବା ପାଇଁ ହୋମ୍ ପେଜ୍ ରେ ସନ୍ଧାନ ବାର୍ ବ୍ୟବହାର କରନ୍ତୁ।
5. **ମାପ୍ ରେ ଅବସ୍ଥାନ ଦେଖନ୍ତୁ**: ପ୍ରତ୍ୟେକ ଫାର୍ମର ଅବସ୍ଥାନ ମାପ୍ ରେ ଦେଖନ୍ତୁ ଓ ଆପଣଙ୍କର କ୍ରୟ ଯୋଜନା କରନ୍ତୁ।
6. **ସହାୟତା ପାଆନ୍ତୁ**: ଯେକଣସି ସହାୟତା ପାଇଁ support@farmart.com ଠାରେ ସମ୍ପର୍କ କରନ୍ତୁ।"""

    elif "i'm" in user_input.lower() or "iam" in user_input.lower() or "my name is" in user_input.lower() or "myself" in user_input.lower():
        # Extracting the name from the user input
        if "i'm" in user_input.lower():
            name_start_index = user_input.lower().find("i'm") + 4
        elif "iam" in user_input.lower():
            name_start_index = user_input.lower().find("iam") + 3
        elif "my name is" in user_input.lower():
            name_start_index = user_input.lower().find("my name is") + 11
        elif "myself" in user_input.lower():
            name_start_index = user_input.lower().find("myself") + 7
    
        name_end_index = user_input.find(" ", name_start_index)
        if name_end_index == -1:
            name = user_input[name_start_index:].strip()
        else:
            name = user_input[name_start_index:name_end_index].strip()
    
        bot_response = f"Hello {name}, welcome to FarMart assistance."
    else:
        bot_response = "I'm not sure about that. Please visit our website at https://farmart.com for more information."

    return bot_response

if __name__ == '__main__':
    app.run(debug=True)
