"""
Gender Text Transformer - Flask Backend for Render
"""

import os
import tempfile
import re
from dataclasses import dataclass
from typing import List, Tuple, Literal

from flask import Flask, request, send_file, jsonify, after_this_request
from flask_cors import CORS
import fitz  # pymupdf


# =============================================================================
# Gender Transformation Logic
# =============================================================================

GENDER_PAIRS = [
    ('he', 'she'), ('him', 'her'), ('himself', 'herself'),
    ('man', 'woman'), ('men', 'women'), ('male', 'female'), ('males', 'females'),
    ('boy', 'girl'), ('boys', 'girls'),
    ('father', 'mother'), ('fathers', 'mothers'), ('dad', 'mom'), ('daddy', 'mommy'),
    ('son', 'daughter'), ('sons', 'daughters'),
    ('brother', 'sister'), ('brothers', 'sisters'),
    ('uncle', 'aunt'), ('uncles', 'aunts'),
    ('nephew', 'niece'), ('nephews', 'nieces'),
    ('husband', 'wife'), ('husbands', 'wives'),
    ('grandfather', 'grandmother'), ('grandfathers', 'grandmothers'),
    ('grandpa', 'grandma'), ('grandson', 'granddaughter'),
    ('stepfather', 'stepmother'), ('stepson', 'stepdaughter'),
    ('stepbrother', 'stepsister'), ('godfather', 'godmother'),
    ('godson', 'goddaughter'),
    ('mr', 'ms'), ('sir', 'madam'),
    ('gentleman', 'lady'), ('gentlemen', 'ladies'),
    ('lord', 'lady'), ('lords', 'ladies'),
    ('king', 'queen'), ('kings', 'queens'),
    ('prince', 'princess'), ('princes', 'princesses'),
    ('duke', 'duchess'), ('baron', 'baroness'),
    ('count', 'countess'), ('emperor', 'empress'),
    ('actor', 'actress'), ('waiter', 'waitress'),
    ('steward', 'stewardess'), ('host', 'hostess'),
    ('hero', 'heroine'), ('heroes', 'heroines'),
    ('god', 'goddess'), ('gods', 'goddesses'),
    ('priest', 'priestess'), ('monk', 'nun'), ('monks', 'nuns'),
    ('wizard', 'witch'), ('wizards', 'witches'),
    ('widower', 'widow'), ('widowers', 'widows'),
    ('bachelor', 'bachelorette'), ('groom', 'bride'),
    ('salesman', 'saleswoman'), ('salesmen', 'saleswomen'),
    ('businessman', 'businesswoman'), ('businessmen', 'businesswomen'),
    ('congressman', 'congresswoman'), ('congressmen', 'congresswomen'),
    ('policeman', 'policewoman'), ('policemen', 'policewomen'),
    ('fireman', 'firewoman'), ('firemen', 'firewomen'),
    ('spokesman', 'spokeswoman'), ('spokesmen', 'spokeswomen'),
    ('chairman', 'chairwoman'), ('chairmen', 'chairwomen'),
    ('boyfriend', 'girlfriend'), ('boyfriends', 'girlfriends'),
    ('manhood', 'womanhood'), ('mankind', 'womankind'),
    ('masculine', 'feminine'), ('masculinity', 'femininity'),
    ('lad', 'lass'), ('lads', 'lasses'),
    ('guy', 'gal'), ('guys', 'gals'),
]


def preserve_case(original: str, replacement: str) -> str:
    if original.isupper():
        return replacement.upper()
    elif original[0].isupper():
        return replacement.capitalize()
    return replacement.lower()


def build_mappings(direction: str) -> dict:
    if direction == 'm_to_f':
        return {m.lower(): f.lower() for m, f in GENDER_PAIRS}
    else:
        return {f.lower(): m.lower() for m, f in GENDER_PAIRS}


def transform_text(text: str, direction: str = 'm_to_f') -> str:
    if not text or not text.strip():
        return text
    
    mappings = build_mappings(direction)
    
    if direction == 'm_to_f':
        text = re.sub(
            r'\bhis\b(?=\s+[a-zA-Z])',
            lambda m: preserve_case(m.group(), 'her'),
            text, flags=re.IGNORECASE
        )
        text = re.sub(
            r'\bhis\b',
            lambda m: preserve_case(m.group(), 'hers'),
            text, flags=re.IGNORECASE
        )
    else:
        text = re.sub(
            r'\bhers\b',
            lambda m: preserve_case(m.group(), 'his'),
            text, flags=re.IGNORECASE
        )
    
    for source, target in mappings.items():
        if source in ('his', 'hers'):
            continue
        
        if source == 'her' and direction == 'f_to_m':
            text = re.sub(
                r'\bher\b(?=\s+[a-zA-Z])',
                lambda m: preserve_case(m.group(), 'his'),
                text, flags=re.IGNORECASE
            )
            text = re.sub(
                r'\bher\b',
                lambda m: preserve_case(m.group(), 'him'),
                text, flags=re.IGNORECASE
            )
            continue
        
        pattern = rf'\b{re.escape(source)}\b'
        text = re.sub(
            pattern,
            lambda m, t=target: preserve_case(m.group(), t),
            text, flags=re.IGNORECASE
        )
    
    return text


# =============================================================================
# PDF Processing (FIXED & OPTIMIZED)
# =============================================================================

def normalize_text(text: str) -> str:
    """
    Fixes the 'dots' issue by replacing unsupported unicode characters 
    with their ASCII equivalents.
    """
    replacements = {
        "\u2018": "'",  # Left single quote
        "\u2019": "'",  # Right single quote (apostrophe)
        "\u201C": '"',  # Left double quote
        "\u201D": '"',  # Right double quote
        "\u2013": "-",  # En dash
        "\u2014": "-",  # Em dash
        "…": "...",     # Ellipsis
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text

def get_text_spans(page):
    spans = []
    # "dict" gives us text, font, size, and bbox
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                spans.append({
                    "text": span["text"],
                    "bbox": fitz.Rect(span["bbox"]),
                    "font": span["font"],
                    "size": span["size"],
                    "color": span["color"],
                    "origin": span["origin"], # This comes as a sequence, we convert to Point later
                })
    return spans

def find_matching_font(original_font: str) -> str:
    # Improved mapping to handle bold/italic better
    original_font = original_font.lower()
    if "bold" in original_font and "times" in original_font: return "tib"
    if "bold" in original_font and "courier" in original_font: return "cob"
    if "bold" in original_font: return "helvb" # Default bold
    if "italic" in original_font and "times" in original_font: return "tii"
    if "italic" in original_font: return "helvi" # Default italic
    
    font_fallbacks = {
        "arial": "helv", "helvetica": "helv",
        "times": "tiro", "roman": "tiro",
        "courier": "cour", "mono": "cour",
    }
    for key, value in font_fallbacks.items():
        if key in original_font:
            return value
    return "helv"

def transform_pdf(input_path: str, output_path: str, direction: str = 'm_to_f') -> dict:
    doc = fitz.open(input_path)
    stats = {"pages_processed": 0, "spans_modified": 0}
    
    # Pre-load base fonts to calculate text length
    font_cache = {} 
    
    for page in doc:
        spans = get_text_spans(page)
        modifications = []
        
        # 1. Identify all modifications needed on this page
        for span in spans:
            original_text = span["text"]
            
            # Apply your regex transformation logic
            try:
                transformed_text = transform_text(original_text, direction) 
            except Exception:
                transformed_text = original_text
            
            # Fix "dots" by normalizing Unicode
            transformed_text = normalize_text(transformed_text)

            if original_text != transformed_text:
                modifications.append({
                    "original": original_text,
                    "transformed": transformed_text,
                    "bbox": span["bbox"],
                    "font": span["font"],
                    "size": span["size"],
                    "color": span["color"],
                    "origin": span["origin"],
                })

        if not modifications:
            continue

        # 2. Redact old text (Process all redactions first to clear the canvas)
        for mod in modifications:
            bbox = mod["bbox"]
            page.add_redact_annot(bbox) 
        
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

        # 3. Insert new text with SCALING (Morphing)
        for mod in modifications:
            fontname = find_matching_font(mod["font"])
            
            # Instantiate font object to calculate width
            if fontname not in font_cache:
                font_cache[fontname] = fitz.Font(fontname)
            font_obj = font_cache[fontname]

            # Calculate Scale Factor
            new_text_width = font_obj.text_length(mod["transformed"], fontsize=mod["size"])
            original_width = mod["bbox"].width
            
            scale_x = 1.0
            if new_text_width > 0:
                scale_x = original_width / new_text_width
            
            # Clamp scaling (prevent extreme stretching)
            scale_x = max(0.6, min(scale_x, 1.5))

            # Extract color
            color_int = mod["color"]
            r = ((color_int >> 16) & 0xFF) / 255.0
            g = ((color_int >> 8) & 0xFF) / 255.0
            b = (color_int & 0xFF) / 255.0

            # Convert origin tuple to Point object (Critical for morph!)
            origin_pt = fitz.Point(mod["origin"])

            try:
                # morph=(origin, matrix) scales the text horizontally
                page.insert_text(
                    origin_pt, 
                    mod["transformed"], 
                    fontname=fontname, 
                    fontsize=mod["size"], 
                    color=(r, g, b),
                    morph=(origin_pt, fitz.Matrix(scale_x, 1))
                )
            except Exception as e:
                print(f"Error inserting text '{mod['transformed']}': {e}")
                # Fallback without morph if it fails
                try:
                    page.insert_text(
                        origin_pt, 
                        mod["transformed"], 
                        fontsize=mod["size"],
                        color=(r, g, b)
                    )
                except:
                    pass

        stats["pages_processed"] += 1
        stats["spans_modified"] += len(modifications)

    doc.save(output_path, garbage=4, deflate=True)
    doc.close()
    return stats


# =============================================================================
# Flask Application
# =============================================================================

app = Flask(__name__)

# Enable CORS for all routes with explicit settings
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"]
    }
})

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max


@app.route('/')
def index():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Gender Text Transformer</title>
  <style>
    html, body { margin: 0; padding: 0; height: 100%; }
    body {
      font-family: sans-serif;
      background-color: #2c2c2c;
      background-image: linear-gradient(135deg, #1a1a1a 0%, #2c2c2c 50%, #1a1a1a 100%);
      min-height: 100vh;
    }
    .tool-container { max-width: 700px; margin: 40px auto; padding: 30px; }
    .tool-section {
      background-color: rgba(255, 255, 255, 0.9);
      border-radius: 12px;
      padding: 30px;
      margin-bottom: 30px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    .tool-section h1 {
      margin-top: 0;
      border-bottom: 3px solid #333;
      padding-bottom: 10px;
      color: #222;
    }
    .intro-text { line-height: 1.6; color: #333; margin-bottom: 25px; }
    .intro-text em { color: maroon; font-style: italic; }
    .form-box {
      margin-bottom: 25px;
      padding: 20px;
      background-color: rgba(240, 240, 240, 0.6);
      border-radius: 8px;
      border-left: 4px solid maroon;
    }
    .form-box h3 { margin-top: 0; color: maroon; margin-bottom: 15px; }
    .form-group { margin-bottom: 20px; }
    .form-group:last-child { margin-bottom: 0; }
    .form-group label { display: block; font-weight: bold; color: #003366; margin-bottom: 8px; }
    input[type="file"] {
      width: 100%;
      padding: 12px;
      border: 2px dashed #999;
      border-radius: 6px;
      background-color: rgba(255, 255, 255, 0.8);
      font-family: inherit;
      font-size: 1rem;
      cursor: pointer;
      box-sizing: border-box;
    }
    input[type="file"]:hover { border-color: maroon; }
    input[type="file"]::file-selector-button {
      padding: 8px 16px;
      margin-right: 12px;
      border: none;
      border-radius: 4px;
      background-color: maroon;
      color: white;
      cursor: pointer;
    }
    input[type="file"]::file-selector-button:hover { background-color: #003366; }
    .radio-group { display: flex; flex-direction: column; gap: 10px; }
    .radio-option {
      display: flex;
      align-items: flex-start;
      padding: 12px 15px;
      background-color: rgba(255, 255, 255, 0.8);
      border: 1px solid #ccc;
      border-radius: 6px;
      cursor: pointer;
    }
    .radio-option:hover { background-color: rgba(255, 255, 255, 0.95); border-color: #999; }
    .radio-option.selected { border-color: maroon; border-left: 4px solid maroon; }
    .radio-option input[type="radio"] { margin-right: 12px; margin-top: 3px; accent-color: maroon; }
    .radio-label .title { color: #222; font-weight: bold; display: block; margin-bottom: 2px; }
    .radio-label .desc { color: #666; font-size: 0.9em; }
    .submit-btn {
      width: 100%;
      padding: 14px 30px;
      border: none;
      border-radius: 6px;
      background-color: maroon;
      color: white;
      font-size: 1.1rem;
      font-weight: bold;
      cursor: pointer;
    }
    .submit-btn:hover { background-color: #003366; }
    .submit-btn:disabled { background-color: #999; cursor: wait; }
    #status { margin-top: 20px; padding: 15px; border-radius: 6px; text-align: center; display: none; }
    #status.loading { background-color: rgba(0, 51, 102, 0.1); border: 1px solid #003366; color: #003366; }
    #status.success { background-color: rgba(0, 128, 0, 0.1); border: 1px solid green; color: green; }
    #status.error { background-color: rgba(128, 0, 0, 0.1); border: 1px solid maroon; color: maroon; }
    .examples-section { margin-top: 25px; padding-top: 20px; border-top: 2px solid #ddd; }
    .examples-section h3 { margin-top: 0; color: #333; font-size: 1em; margin-bottom: 15px; }
    .examples-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 20px; }
    .example-item { color: #555; font-size: 0.95em; }
    .example-item span { color: maroon; font-weight: 500; }
    .back-link { display: block; text-align: center; margin-top: 20px; color: #ccc; text-decoration: none; }
    .back-link:hover { color: white; }
    @media (max-width: 600px) {
      .tool-container { margin: 20px; padding: 15px; }
      .examples-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="tool-container">
    <div class="tool-section">
      <h1>Gender Text Transformer</h1>
      <p class="intro-text">
        Upload a PDF document to transform gendered language while preserving 
        the original formatting. This tool performs systematic substitution of 
        gendered terms—<em>he</em> becomes <em>she</em>, <em>father</em> becomes 
        <em>mother</em>, and so forth—allowing for experimental readings and 
        critical engagement with texts.
      </p>
      <form id="uploadForm">
        <div class="form-box">
          <h3>Upload Document</h3>
          <div class="form-group">
            <label for="file">Select PDF File</label>
            <input type="file" id="file" accept=".pdf" required>
          </div>
          <div class="form-group">
            <label>Transformation Mode</label>
            <div class="radio-group">
              <label class="radio-option selected">
                <input type="radio" name="direction" value="m_to_f" checked>
                <div class="radio-label">
                  <span class="title">Masculine → Feminine</span>
                  <span class="desc">he, him, father, king → she, her, mother, queen</span>
                </div>
              </label>
              <label class="radio-option">
                <input type="radio" name="direction" value="f_to_m">
                <div class="radio-label">
                  <span class="title">Feminine → Masculine</span>
                  <span class="desc">she, her, mother, queen → he, him, father, king</span>
                </div>
              </label>
              <label class="radio-option">
                <input type="radio" name="direction" value="swap">
                <div class="radio-label">
                  <span class="title">Swap Both</span>
                  <span class="desc">Exchange all gendered terms bidirectionally</span>
                </div>
              </label>
            </div>
          </div>
        </div>
        <button type="submit" class="submit-btn" id="submitBtn">Transform Document</button>
      </form>
      <div id="status"></div>
      <div class="examples-section">
        <h3>Sample Transformations</h3>
        <div class="examples-grid">
          <div class="example-item">he/him/his → <span>she/her/hers</span></div>
          <div class="example-item">Mr., Sir → <span>Ms., Madam</span></div>
          <div class="example-item">father, son → <span>mother, daughter</span></div>
          <div class="example-item">king, prince → <span>queen, princess</span></div>
          <div class="example-item">brother, uncle → <span>sister, aunt</span></div>
          <div class="example-item">husband, groom → <span>wife, bride</span></div>
        </div>
      </div>
    </div>
    <a href="https://criticaltheoryclub.neocities.org/sessions" class="back-link">← Back to Critical Theory Reading Group</a>
  </div>
  <script>
    document.querySelectorAll('.radio-option').forEach(option => {
      option.addEventListener('click', () => {
        document.querySelectorAll('.radio-option').forEach(o => o.classList.remove('selected'));
        option.classList.add('selected');
      });
    });
    document.getElementById('uploadForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const fileInput = document.getElementById('file');
      const direction = document.querySelector('input[name="direction"]:checked').value;
      const status = document.getElementById('status');
      const btn = document.getElementById('submitBtn');
      if (!fileInput.files[0]) { showStatus('error', 'Please select a file.'); return; }
      if (fileInput.files[0].size > 50 * 1024 * 1024) { showStatus('error', 'File exceeds 50MB limit.'); return; }
      btn.disabled = true;
      btn.textContent = 'Processing...';
      if (direction === 'swap') {
        showStatus('loading', 'Performing bidirectional swap (step 1 of 2)...');
        try {
          let formData = new FormData();
          formData.append('file', fileInput.files[0]);
          formData.append('direction', 'm_to_f');
          let response = await fetch('/transform', { method: 'POST', body: formData });
          if (!response.ok) throw new Error('First transformation failed');
          let blob = await response.blob();
          showStatus('loading', 'Performing bidirectional swap (step 2 of 2)...');
          formData = new FormData();
          formData.append('file', new File([blob], 'temp.pdf', { type: 'application/pdf' }));
          formData.append('direction', 'f_to_m');
          response = await fetch('/transform', { method: 'POST', body: formData });
          if (!response.ok) throw new Error('Second transformation failed');
          blob = await response.blob();
          downloadBlob(blob, 'swapped_' + fileInput.files[0].name);
          showStatus('success', 'Transformation complete. Download started.');
          fileInput.value = '';
        } catch (err) { showStatus('error', err.message); }
        finally { btn.disabled = false; btn.textContent = 'Transform Document'; }
        return;
      }
      showStatus('loading', 'Transforming document...');
      const formData = new FormData();
      formData.append('file', fileInput.files[0]);
      formData.append('direction', direction);
      try {
        const response = await fetch('/transform', { method: 'POST', body: formData });
        if (!response.ok) { let errorMsg = 'Transformation failed'; try { const err = await response.json(); errorMsg = err.error || errorMsg; } catch {} throw new Error(errorMsg); }
        const blob = await response.blob();
        downloadBlob(blob, 'transformed_' + fileInput.files[0].name);
        showStatus('success', 'Transformation complete. Download started.');
        fileInput.value = '';
      } catch (err) { showStatus('error', err.message); }
      finally { btn.disabled = false; btn.textContent = 'Transform Document'; }
    });
    function showStatus(type, message) { const status = document.getElementById('status'); status.style.display = 'block'; status.className = type; status.textContent = message; }
    function downloadBlob(blob, filename) { const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; a.download = filename; document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url); }
  </script>
</body>
</html>'''


@app.route('/api')
def api_info():
    return jsonify({
        "service": "Gender Text Transformer",
        "endpoints": {
            "/transform": "POST - Upload PDF and get transformed PDF back",
            "/health": "GET - Health check"
        }
    })


@app.route('/health')
def health():
    return jsonify({"status": "ok"})


@app.route('/transform', methods=['POST'])
def transform():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'File must be a PDF'}), 400
    
    direction = request.form.get('direction', 'm_to_f')
    if direction not in ('m_to_f', 'f_to_m'):
        return jsonify({'error': 'Invalid direction. Use m_to_f or f_to_m'}), 400
    
    input_path = None
    output_path = None
    
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_in:
            file.save(tmp_in.name)
            input_path = tmp_in.name
        
        # Create output file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_out:
            output_path = tmp_out.name
        
        # Transform (Now using the fixed function)
        stats = transform_pdf(input_path, output_path, direction)
        
        # Clean up input immediately
        if input_path and os.path.exists(input_path):
            os.unlink(input_path)
            input_path = None

        # Clean up output after request is sent
        @after_this_request
        def remove_file(response):
            try:
                if output_path and os.path.exists(output_path):
                    os.unlink(output_path)
            except Exception as e:
                print(f"Error cleaning up output file: {e}")
            return response

        # Send result
        return send_file(
            output_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'transformed_{file.filename}'
        )
        
    except Exception as e:
        # If something went wrong, clean up now
        if input_path and os.path.exists(input_path):
            os.unlink(input_path)
        if output_path and os.path.exists(output_path):
            os.unlink(output_path)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
