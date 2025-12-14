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

def get_text_spans(page):
    spans = []
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
                    "origin": span["origin"],
                })
    return spans


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


def transform_pdf(input_path: str, output_path: str, direction: str = 'm_to_f') -> dict:
    doc = fitz.open(input_path)
    stats = {"pages_processed": 0, "spans_modified": 0, "words_changed": []}
    
    for page in doc:
        spans = get_text_spans(page)
        modifications = []
        
        for span in spans:
            original_text = span["text"]
            transformed_text = transform_text(original_text, direction)
            
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
        
        for mod in reversed(modifications):
            bbox = mod["bbox"]
            redact_rect = fitz.Rect(bbox.x0 - 0.5, bbox.y0 - 0.5, bbox.x1 + 0.5, bbox.y1 + 0.5)
            annot = page.add_redact_annot(redact_rect)
            annot.set_colors(stroke=(1, 1, 1), fill=(1, 1, 1))
            stats["spans_modified"] += 1
        
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
        
        for mod in modifications:
            fontname = find_matching_font(mod["font"])
            color_int = mod["color"]
            r = ((color_int >> 16) & 0xFF) / 255.0
            g = ((color_int >> 8) & 0xFF) / 255.0
            b = (color_int & 0xFF) / 255.0
            
            try:
                page.insert_text(
                    mod["origin"], mod["transformed"],
                    fontname=fontname, fontsize=mod["size"], color=(r, g, b)
                )
            except:
                try:
                    page.insert_text(mod["origin"], mod["transformed"], fontsize=mod["size"])
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
CORS(app)  # Enable CORS for all routes
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max


@app.route('/')
def index():
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
