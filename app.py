"""
Gender Text Transformer - Flask Backend for Render
"""

import os
import tempfile
import re
from dataclasses import dataclass
from typing import List, Tuple, Literal

from flask import Flask, request, send_file, jsonify
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
# PDF Processing
# =============================================================================

def normalize_text(text: str) -> str:
    """Normalize special characters to ASCII equivalents for better font compatibility."""
    replacements = {
        ''': "'", ''': "'", '"': '"', '"': '"',
        '–': '-', '—': '-', '…': '...',
        'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬀ': 'ff', 'ﬃ': 'ffi', 'ﬄ': 'ffl',
        '′': "'", '″': '"',
        '\u00a0': ' ', '\u2019': "'", '\u2018': "'",
        '\u201c': '"', '\u201d': '"',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def get_font_info_at_point(page, point):
    """Get font information at a specific point on the page."""
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                bbox = fitz.Rect(span["bbox"])
                if bbox.contains(point):
                    return {
                        "font": span["font"],
                        "size": span["size"],
                        "color": span["color"],
                        "origin": span["origin"],
                    }
    # Default fallback
    return {"font": "Helvetica", "size": 11, "color": 0, "origin": point}


def find_matching_font(original_font: str) -> str:
    font_fallbacks = {
        "Arial": "helv", "Helvetica": "helv",
        "Times": "tiro", "Times New Roman": "tiro",
        "Courier": "cour", "CourierNew": "cour",
    }
    for key, value in font_fallbacks.items():
        if key.lower() in original_font.lower():
            return value
    return "helv"


def build_word_list(direction: str) -> list:
    """Build list of (source_word, target_word) tuples to search for."""
    words_to_find = []
    mappings = build_mappings(direction)
    
    # Add all words from mappings
    for source, target in mappings.items():
        if source not in ('his', 'hers', 'her'):  # Handle these specially
            words_to_find.append((source, target))
    
    # Handle possessives specially based on direction
    if direction == 'm_to_f':
        # "his" can become "her" or "hers" depending on context
        # We'll just use "her" as it's more common (possessive adjective)
        words_to_find.append(('his', 'her'))
    else:
        words_to_find.append(('hers', 'his'))
        words_to_find.append(('her', 'his'))  # This is imperfect but workable
    
    return words_to_find


def transform_pdf(input_path: str, output_path: str, direction: str = 'm_to_f') -> dict:
    doc = fitz.open(input_path)
    stats = {"pages_processed": 0, "spans_modified": 0, "words_changed": []}
    
    words_to_find = build_word_list(direction)
    
    for page in doc:
        replacements = []  # List of (rect, original, replacement, font_info)
        
        # Search for each word that needs replacing
        for source_word, target_word in words_to_find:
            # Search for different case variants
            variants = [
                (source_word.lower(), target_word.lower()),
                (source_word.capitalize(), target_word.capitalize()),
                (source_word.upper(), target_word.upper()),
            ]
            
            for source_variant, target_variant in variants:
                # Find all instances of this word on the page
                rects = page.search_for(source_variant, quads=False)
                
                for rect in rects:
                    # Verify it's a whole word by checking characters before/after
                    # Get text in a slightly expanded area
                    expanded = fitz.Rect(rect.x0 - 10, rect.y0, rect.x1 + 10, rect.y1)
                    surrounding_text = page.get_text("text", clip=expanded)
                    
                    # Simple word boundary check
                    # Find position of our word in surrounding text
                    idx = surrounding_text.lower().find(source_variant.lower())
                    if idx != -1:
                        # Check character before
                        if idx > 0:
                            char_before = surrounding_text[idx - 1]
                            if char_before.isalnum():
                                continue  # Part of a larger word
                        # Check character after
                        end_idx = idx + len(source_variant)
                        if end_idx < len(surrounding_text):
                            char_after = surrounding_text[end_idx]
                            if char_after.isalnum():
                                continue  # Part of a larger word
                    
                    # Get font info from this location
                    center_point = fitz.Point(rect.x0 + 1, (rect.y0 + rect.y1) / 2)
                    font_info = get_font_info_at_point(page, center_point)
                    
                    replacements.append({
                        "rect": rect,
                        "original": source_variant,
                        "replacement": normalize_text(target_variant),
                        "font_info": font_info,
                    })
        
        # Remove duplicates (same rect)
        seen_rects = set()
        unique_replacements = []
        for r in replacements:
            rect_key = (round(r["rect"].x0, 1), round(r["rect"].y0, 1), 
                       round(r["rect"].x1, 1), round(r["rect"].y1, 1))
            if rect_key not in seen_rects:
                seen_rects.add(rect_key)
                unique_replacements.append(r)
        
        # Apply redactions
        for r in unique_replacements:
            rect = r["rect"]
            font_size = r["font_info"]["size"]
            
            # Padding based on font size
            v_padding = font_size * 0.3
            h_padding = 2.0
            
            redact_rect = fitz.Rect(
                rect.x0 - h_padding,
                rect.y0 - v_padding,
                rect.x1 + h_padding,
                rect.y1 + v_padding
            )
            annot = page.add_redact_annot(redact_rect)
            annot.set_colors(stroke=(1, 1, 1), fill=(1, 1, 1))
            stats["spans_modified"] += 1
        
        # Apply all redactions at once
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
        
        # Insert replacement text
        for r in unique_replacements:
            font_info = r["font_info"]
            fontname = find_matching_font(font_info["font"])
            
            color_int = font_info["color"]
            if isinstance(color_int, int):
                red = ((color_int >> 16) & 0xFF) / 255.0
                green = ((color_int >> 8) & 0xFF) / 255.0
                blue = (color_int & 0xFF) / 255.0
            else:
                red, green, blue = 0, 0, 0
            
            # Calculate insertion point (bottom-left of rect, adjusted for baseline)
            rect = r["rect"]
            # Use the baseline from font_info if available, otherwise estimate
            if "origin" in font_info and font_info["origin"]:
                insertion_point = fitz.Point(rect.x0, font_info["origin"][1])
            else:
                # Estimate baseline as ~80% down from top of rect
                baseline_y = rect.y0 + (rect.height * 0.8)
                insertion_point = fitz.Point(rect.x0, baseline_y)
            
            try:
                page.insert_text(
                    insertion_point,
                    r["replacement"],
                    fontname=fontname,
                    fontsize=font_info["size"],
                    color=(red, green, blue),
                    encoding=fitz.TEXT_ENCODING_UTF8
                )
                stats["words_changed"].append((r["original"], r["replacement"]))
            except Exception as e:
                # Fallback with simpler parameters
                try:
                    page.insert_text(
                        insertion_point,
                        r["replacement"],
                        fontname="helv",
                        fontsize=font_info["size"]
                    )
                except:
                    pass
        
        stats["pages_processed"] += 1
    
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
        
        # Transform
        stats = transform_pdf(input_path, output_path, direction)
        
        # Send result
        return send_file(
            output_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'transformed_{file.filename}'
        )
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Cleanup
        if input_path and os.path.exists(input_path):
            try:
                os.unlink(input_path)
            except:
                pass
        # Note: output_path cleanup happens after send_file completes


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
