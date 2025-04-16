#Run this code to download the models locally
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

# Save TrOCR model
trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
trocr_processor.save_pretrained("./models/trocr")
trocr_model.save_pretrained("./models/trocr")

# Save PEGASUS model
pegasus_tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
pegasus_model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")
pegasus_tokenizer.save_pretrained("./models/pegasus")
pegasus_model.save_pretrained("./models/pegasus")
