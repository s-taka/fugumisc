# -*- coding: utf-8 -*-

# This script depends on pymupdf (AGPL: GNU Affero General Public License)

import pickle
import argparse
import json
import logging
from datetime import datetime

import time
import sys
import os
import sqlite3
import pprint
import base64
import re
import html

import subprocess
import psutil
import functools

from subprocess import PIPE
from subprocess import TimeoutExpired

from io import StringIO
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

from fugumt.tojpn import FuguJPNTranslator
from fugumt.tojpn import get_err_translated

from fugumt.misc import make_marian_process
from fugumt.misc import close_marian_process
from fugumt.misc import ckeck_restart_marian_process

# for ocr
import pdf2image
import numpy as np
import layoutparser as lp
import torchvision.ops.boxes as bops
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO

import gzip
import fitz
from layoutparser.elements import TextBlock
from layoutparser.elements import Rectangle


def has_intersect(a, b):
  return max(a.x_1, b.x_1) <= min(a.x_2, b.x_2) and max(a.y_1, b.y_1) <= min(a.y_2, b.y_2)

def get_intersect_bound(a, b):
  x1 = max(a.x_1, b.x_1)
  y1 = max(a.y_1, b.y_1)
  x2 = min(a.x_2, b.x_2)
  y2 = min(a.y_2, b.y_2)

  return x1, y1, x2, y2  

def merge_block(block_1, block_2):
  if has_intersect(block_1.block, block_2.block):
    x_1, y_1, x_2, y_2 = get_intersect_bound(block_1.block, block_2.block)
    intersect_area = (x_2 - x_1) * (y_2 - y_1)
    block1_area  = (block_1.block.x_2 - block_1.block.x_1) * (block_1.block.y_2 - block_1.block.y_1)
    block2_area  = (block_2.block.x_2 - block_2.block.x_1) * (block_2.block.y_2 - block_2.block.y_1)
    if block1_area > block2_area:
      if intersect_area / block2_area > 0.9:
         block_2.set(type='None', inplace= True)

def ret_intersect_rate(block_1, block_2):
  if has_intersect(block_1.block, block_2.block):
    x_1, y_1, x_2, y_2 = get_intersect_bound(block_1.block, block_2.block)
    intersect_area = (x_2 - x_1) * (y_2 - y_1)
    block1_area  = (block_1.block.x_2 - block_1.block.x_1) * (block_1.block.y_2 - block_1.block.y_1)
    if block1_area > 0.0:
      return intersect_area / block1_area
    else:
      return 0.0
  return 0.0


def ocr_pdf(pdf_file, logger=None, max_page = 1000000):
    fitz_page_info = []
    fitz_pdf_doc = fitz.open(pdf_file)
    for page in fitz_pdf_doc:
        blocks = page.get_text("blocks", sort=True)
        page_width = page.rect[2] - page.rect[0]
        page_height = page.rect[3] - page.rect[1]
        fitz_page_info.append((page_width, page_height, blocks))


    pdf_images = pdf2image.convert_from_path(pdf_file)
    ocr_agent = lp.TesseractAgent(languages='eng')
    model = lp.models.Detectron2LayoutModel('lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
                                extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.4],
                                label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
    return_blocks = []
    fitz_block_idx = 100000
    for page_idx, pdf_image in enumerate(pdf_images):
        ocr_img = np.asarray(pdf_image)

        layout_result = model.detect(ocr_img)
        for layout_i in layout_result:
            for layout_j in layout_result:
                if layout_i != layout_j:
                    merge_block(layout_i, layout_j)

        layout_blocks = lp.Layout([b for b in layout_result if b.type!='None'])
        text_blocks = lp.Layout([b for b in layout_result if b.type=='Text' or b.type=='List' or b.type=='Title'])
        text_blocks_ocr = []

        fitz_detect_texts = len(fitz_page_info[page_idx][2])
        lp_detext_blocks = len(text_blocks)
        try:
            detect_rate = fitz_detect_texts / lp_detext_blocks
        except:
            detect_rate = 1.0
        if logger:
            logger.info("page_id = {}: {} / {} = {}".format(page_idx, fitz_detect_texts, lp_detext_blocks, detect_rate))

        if (detect_rate > 0.5 or fitz_detect_texts > 5):
            if logger:
              logger.info('pymupdf mode')
            tmp_blocks = []
            (page_width, page_height, fitz_block) = fitz_page_info[page_idx]
            image_width = pdf_image.width
            image_height = pdf_image.height
            scale_width = image_width / page_width
            scale_height = image_height / page_height
            for i, b in enumerate(fitz_block):
                rect = Rectangle(x_1=b[0]*scale_width, y_1=b[1]*scale_height, x_2=b[2]*scale_width, y_2=b[3]*scale_height)
                tb_type = 'Text'
                intersect_list = sorted([(ret_intersect_rate(TextBlock(rect, text=None), lb), j , lb) for j, lb in enumerate(layout_blocks)], reverse=True)
                if len(intersect_list) > 0:
                    max_intersect = intersect_list[0]
                    if max_intersect[0] > 0.8:
                        tb_type = max_intersect[2].type

                # imageも除外(add and rename tmp_blocks)
                if tb_type != 'Table' and tb_type != 'Figure' and not re.search(r'^<image[^>]+>$', b[4]):
                    tb = TextBlock(rect, text=b[4], id='{}'.format(fitz_block_idx), type=tb_type)
                    fitz_block_idx += 1
                    tmp_blocks.append(tb)

            #行分離の対応(add)
            if len(tmp_blocks) > 0:
                prev_block = tmp_blocks[0]
                prev_height = prev_block.block.y_1 - prev_block.block.y_2
            for tb_idx, tb in enumerate(tmp_blocks):
                if tb_idx == 0:
                    continue
                if (abs(prev_block.block.x_2 - tb.block.x_2) < 10.0 or abs(prev_block.block.x_1 - tb.block.x_1) < 10.0) and abs(prev_height - (tb.block.y_1 - tb.block.y_2)) < 10.0:
                    prev_block.block.x_1 = min(prev_block.block.x_1, tb.block.x_1)
                    prev_block.block.x_2 = max(prev_block.block.x_2, tb.block.x_2)
                    prev_block.block.y_2 = tb.block.y_2
                    prev_block.text += tb.text
                    prev_block.text = re.sub(r'(\r?\n)+', '\n', prev_block.text)
                else:
                    text_blocks_ocr.append(prev_block)
                    prev_block = tb
                    prev_height = prev_block.block.y_1 - prev_block.block.y_2
            text_blocks_ocr.append(prev_block)
        else:
            if logger:
              logger.info('tesseract mode')

            for layout_i in text_blocks:
                for layout_j in text_blocks:
                    if layout_i != layout_j:
                        merge_block(layout_i, layout_j)
            
            text_blocks_ocr = lp.Layout([b for b in text_blocks if b.type=='Text' or b.type=='List' or b.type=='Title'])

            for block in text_blocks_ocr:
                segment_image = (block
                                    .pad(left=15, right=15, top=5, bottom=5)
                                    .crop_image(ocr_img))
                text = ocr_agent.detect(segment_image)
                block.set(text=text, inplace=True)

        return_blocks.append((page_idx, pdf_image, text_blocks_ocr, text_blocks))

    return return_blocks

def get_title_abstract(in_data, fgmt, make_marian_conf=None, logger=None):
    ocr_result = in_data['ocr_result']
    translated_blocks = in_data['translated_blocks']

    # 最初のタイトルの自動抽出
    first_page = ocr_result[0][2]
    title_blocks = sorted([b for b in first_page if b.type=='Title'], key=lambda x: x.block.y_1)
    title = ''
    if len(title_blocks) > 0:
        title = title_blocks[0].text
    title = re.sub(r'[^a-zA-Z0-9 \|\@\!\"\'\`\*\+\-\)\(\[\]\{\}\<\>\_\~\=\#\$\%\&\.\,\;\:]', '', title)

    def sort_func(lhs, rhs):
        threshold_xdiff_range = max(lhs['box_info'][2] - lhs['box_info'][0], rhs['box_info'][2] - rhs['box_info'][0]) / 1.5
        if abs(lhs['box_info'][0] - rhs['box_info'][0]) < threshold_xdiff_range:
            return lhs['box_info'][1] - rhs['box_info'][1]
        else:
            return lhs['box_info'][0] - rhs['box_info'][0]

    abstract_strs = []
    for page_num, blocks in enumerate(translated_blocks):
        if page_num > 1:
            break
        sorted_block = sorted(blocks, key=functools.cmp_to_key(sort_func))
        for block in sorted_block:
            block_id = block['block_id']
            coords = block['box_info']
            block_data = {'block_id': int(block_id), 'texts': [], 'coords':coords }
            translated_block = block['translated']
            for translated in translated_block:
                en = translated["en"]
                if len(en) > 100:
                    abstract_strs.append(en)

    from transformers import PegasusForConditionalGeneration, PegasusTokenizer
    import torch
    import time

    src_txt = '.'.join(abstract_strs)[0:min(1024, len('.'.join(abstract_strs)))]
    src_txt = re.sub('\.\s*\.', '.', src_txt)

    if logger:
        logger.info("abstract text [{}]".format(src_txt))

    model_name = 'google/pegasus-cnn_dailymail'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

    start = time.time()
    with torch.no_grad():
        batch = tokenizer.prepare_seq2seq_batch([src_txt], truncation=True, padding='longest', return_tensors="pt").to(
            torch_device)
        translated = model.generate(**batch)
        ret_txt = tokenizer.batch_decode(translated, skip_special_tokens=True)

    marian_processes = []
    if make_marian_conf:
        marian_processes = make_marian_process(make_marian_conf["marian_command"],
                                               make_marian_conf["marian_args_pdf_translator"],
                                               make_marian_conf["pdf_ports"])
    
    if logger:
        logger.info("translate abstract {}".format(ret_txt))

    txt = ret_txt[0]
    if len(src_txt) < 150:
        txt = src_txt
    to_translate = pre_proc_text(txt.replace('<n>', '\n\n'))
    translated = ''
    if len(to_translate):
        translated = fgmt.translate_text(to_translate)
        if fgmt.detected_marian_err:
            translated = ''

    to_translate = pre_proc_text(title.replace('\n\n', '\n'))
    title_ja = ''
    if len(to_translate):
        title_ja = fgmt.translate_text(title)
        if fgmt.detected_marian_err:
            title_ja = ''
    close_marian_process(marian_processes)

    return title, title_ja, translated



def pdf_translate_ocr(pdf_path, fgmt, make_marian_conf=None, logger=None):
    ocr_result = ocr_pdf(pdf_path, max_page=300, logger=logger)
    text_block_id = 0
    translated_blocks = []

    marian_processes = []
    if make_marian_conf:
        marian_processes = make_marian_process(make_marian_conf["marian_command"],
                                               make_marian_conf["marian_args_pdf_translator"],
                                               make_marian_conf["pdf_ports"])
    
    for (page_idx, pdf_image, text_blocks_ocr, text_blocks) in ocr_result:
        page_translated_blocks = []
        for (txt, block_info) in [(b.text, b.block) for b in text_blocks_ocr]:
            retry_max = 3
            translated = None
            marian_processes = ckeck_restart_marian_process(marian_processes, make_marian_conf["max_marian_memory"], make_marian_conf["marian_command"], make_marian_conf["marian_args_pdf_translator"], make_marian_conf["pdf_ports"], logger=logger)
            for i in range(retry_max):
                if logger:
                    logger.info("translate page={} block_id={}".format(page_idx, text_block_id))
                to_translate = pre_proc_text(txt.replace('\n\n', '\n'))
                translated = fgmt.translate_text(to_translate)
                if not fgmt.detected_marian_err:
                    block_data = {'original': txt, 'translated': translated, 'block_id': text_block_id, 'box_info':(block_info.x_1, block_info.y_1, block_info.x_2, block_info.y_2)}
                    page_translated_blocks.append(block_data)
                    text_block_id += 1
                    break
                else:
                    translated = None
                    close_marian_process(marian_processes)
                    marian_processes = make_marian_process(make_marian_conf["marian_command"],
                                                            make_marian_conf["marian_args_pdf_translator"],
                                                            make_marian_conf["pdf_ports"])

                    fgmt.detected_marian_err = False
                    if logger:
                        logger.info(fgmt.get_and_clear_logs())
                        logger.warning("recovery marian processes {}/{}".format(i, retry_max-1))
            if translated is None:
                block_data = {'original': txt, 'translated': get_err_translated(), 'block_id': text_block_id, 'box_info':(block_info.x_1, block_info.y_1, block_info.x_2, block_info.y_2)}
                page_translated_blocks.append(block_data)
            marian_processes = ckeck_restart_marian_process(marian_processes, make_marian_conf["max_marian_memory"],
                                                            make_marian_conf["marian_command"],
                                                            make_marian_conf["marian_args_pdf_translator"],
                                                            make_marian_conf["pdf_ports"],
                                                            logger=logger)
            if logger:
                logger.info(fgmt.get_and_clear_logs())
        translated_blocks.append(page_translated_blocks)

    if make_marian_conf:
        close_marian_process(marian_processes)
    
    return {'ocr_result':ocr_result, 'translated_blocks': translated_blocks}






def escape_break_word(txt):
    return re.sub('([a-zA-Z0-9\/\.\-\:\%\-\~\\\*\"\'\&\$\#\(\)\?\_\,,\@]{100,}?)', "\\1 ", html.escape(txt))


def pre_proc_text(txt):
    if txt:
        ret = txt.replace('i.e.', 'i e ')
        ret = txt.replace('e.g.', 'e g ')
        ret = ret.replace('et al.', 'et al ')
        ret = ret.replace('state of the art', 'state-of-the-art')
        ret = ret.replace(' Fig.', ' Fig ')
        ret = ret.replace(' fig.', ' fig ')
        ret = ret.replace(' cf. ', ' cf ')
        ret = ret.replace(' Eq.', ' Eq ')
        ret = ret.replace(' Appx.', ' Appx ')
        ret = re.sub(r'^Fig. ', 'Fig ', ret)
        ret = re.sub(r'^fig. ', 'fig ', ret)
        ret = re.sub(r'^Eq. ', 'Eq ', ret)       
    else:
        ret = txt
    return ret


def pdf_translate(pdf_path, fgmt, make_marian_conf=None, logger=None):
    page_split_tag = '\n\n<<PAGE_SPLIT_TAG>>\n\n'
    output_string = StringIO()
    with open(pdf_path, 'rb') as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams(boxes_flow=0.3, line_margin=1.0))
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for idx, page in enumerate(PDFPage.create_pages(doc)):
            interpreter.process_page(page)
            output_string.write(page_split_tag)
    pdf_text = output_string.getvalue()
    pdf_pages = pdf_text.split(page_split_tag)
    marian_processes = []
    if make_marian_conf:
        marian_processes = make_marian_process(make_marian_conf["marian_command"],
                                               make_marian_conf["marian_args_pdf_translator"],
                                               make_marian_conf["pdf_ports"])
    ret = []
    for pdf_idx, pdf_page in enumerate(pdf_pages[:-1]):
        retry_max = 3
        translated = None
        marian_processes = ckeck_restart_marian_process(marian_processes, make_marian_conf["max_marian_memory"], make_marian_conf["marian_command"], make_marian_conf["marian_args_pdf_translator"], make_marian_conf["pdf_ports"], logger=logger)
        for i in range(retry_max):
            if logger:
                logger.info("translate page={}".format(pdf_idx))
            to_translate = pre_proc_text(pdf_page)
            translated = fgmt.translate_text(to_translate)
            if not fgmt.detected_marian_err:
                ret.append(translated)
                break
            else:
                translated = None
                close_marian_process(marian_processes)
                marian_processes = make_marian_process(make_marian_conf["marian_command"],
                                                       make_marian_conf["marian_args_pdf_translator"],
                                                       make_marian_conf["pdf_ports"])

                fgmt.detected_marian_err = False
                if logger:
                    logger.info(fgmt.get_and_clear_logs())
                    logger.warning("recovery marian processes {}/{}".format(i, retry_max-1))
        if translated is None:
            ret.append(get_err_translated())
        marian_processes = ckeck_restart_marian_process(marian_processes, make_marian_conf["max_marian_memory"],
                                                        make_marian_conf["marian_command"],
                                                        make_marian_conf["marian_args_pdf_translator"],
                                                        make_marian_conf["pdf_ports"],
                                                        logger=logger)
        if logger:
            logger.info(fgmt.get_and_clear_logs())

    if make_marian_conf:
        close_marian_process(marian_processes)

    return ret


def do_db(db_file, pdf_dir, pickle_dir, fgmt, make_marian_conf=None, logger=None, template=""):
    pdf_name = ""
    status = ""
    try:
        with sqlite3.connect(db_file, timeout=120) as db_con:
            ret = db_con.execute('SELECT pdf_path_name, status FROM status WHERE status=? order by date_str;',
                                 ('uploaded',)).fetchone()
            if ret:
                pdf_name = ret[0]
                status = ret[1]
    except:
        return pprint.pformat(sys.exc_info())

    if len(pdf_name) > 0:
        try:
            ret_str = "process [{}]\n".format(pdf_name)
            with sqlite3.connect(db_file) as db_con:
                ret = db_con.execute('UPDATE status set status=?, date_str=? WHERE pdf_path_name=?;',
                                     ('translate to jpn', datetime.today(), pdf_name)).fetchone()
                db_con.commit()
            pdf_path = os.path.join(pdf_dir, pdf_name)
            ret = pdf_translate(pdf_path, fgmt, make_marian_conf=make_marian_conf, logger=logger)

            with open(os.path.join(pickle_dir, pdf_name + ".pickle"), "wb") as f:
                pickle.dump(ret, f)

            with sqlite3.connect(db_file, timeout=120) as db_con:
                ret = db_con.execute('UPDATE status set status=?, date_str=? WHERE pdf_path_name=?;',
                                     ('complete', datetime.today(), pdf_name)).fetchone()
                db_con.commit()
            ret_str += fgmt.get_and_clear_logs()

            if len(template):
                make_static_html(pdf_path, os.path.join(pickle_dir, pdf_name + ".pickle"),
                                 pdf_path + ".html", template=template)
            return ret_str
        except:
            with sqlite3.connect(db_file, timeout=120) as db_con:
                ret = db_con.execute('UPDATE status set status=? WHERE pdf_path_name=?;',
                                     ('error', pdf_name)).fetchone()
                db_con.commit()
            return pprint.pformat(sys.exc_info())
    return "Nothing to do."


def make_static_html(pdf_path, pickle_path, html_path, template="template/pdf_server_static.tmpl", add_data=""):
    with open(template, encoding="utf-8") as in_file:
        tmpl = in_file.read()

    with open(pdf_path, "rb") as in_pdf:
        pdf_base64 = base64.b64encode(in_pdf.read()).decode("utf-8")

    table_header_tmpl = "<div id='translated_{}'><table border='1'><tr><th>英語</th><th>日本語</th><th>スコア</th></tr>\n"
    table_footer_tmpl = "</table></div>\n"
    tr_tmpl = "<tr> <td>{}</td> <td>{}</td> <td>{:.2f}</td></tr>\n"
    tr_tmpl_parse = "<tr> <td>{}</td> <td>{} <br /><small>訳抜け防止モード: {}</small></td> <td>{:.2f}</td></tr>\n"

    pickle_data = pickle.load(open(pickle_path, "rb"))
    translated_tables = ""
    for page_num, translated_page in enumerate(pickle_data):
        translated_tables += table_header_tmpl.format(page_num+1)
        add_item = {"en": "", "ja_best": "", "ja_norm": "", "scores": []}
        for translated in translated_page:
            best_is_norm = 1
            add_item["scores"].append(translated["ja_best_score"])
            add_item["en"] += translated["en"]
            add_item["ja_best"] += translated["ja_best"]
            add_item["ja_norm"] += translated["ja_norm"]
            if translated["best_is_norm"] == 0:
                best_is_norm = 0
            if len(add_item["ja_best"]) < 10:
                continue
            show_score = sum(add_item["scores"]) / len(add_item["scores"])
            if best_is_norm == 1:
                translated_tables += tr_tmpl.format(escape_break_word(add_item["en"]),
                                                    escape_break_word(add_item["ja_best"]), show_score)
            else:
                translated_tables += tr_tmpl_parse.format(escape_break_word(add_item["en"]),
                                                          escape_break_word(add_item["ja_norm"]),
                                                          escape_break_word(add_item["ja_best"]),
                                                          show_score)
            add_item = {"en": "", "ja_best": "", "ja_norm": "", "scores": []}
        if len(add_item["en"]):
            show_score = sum(add_item["scores"]) / len(add_item["scores"])
            translated_tables += tr_tmpl.format(escape_break_word(add_item["en"]),
                                                escape_break_word(add_item["ja_best"]), show_score)
        translated_tables += table_footer_tmpl

    page_list_tmpl = "<button id='nav_{}' onclick='renderPage({})'>{}</button>\n"
    page_list = "&nbsp;".join([page_list_tmpl.format(idx+1, idx+1, idx+1) for idx in range(len(pickle_data))])

    with open(html_path, "w") as out:
        write_data = tmpl.replace("{{translated_tables}}", translated_tables)
        write_data = write_data.replace("{{navigation}}", page_list)
        write_data = write_data.replace("{{base64_pdf}}", pdf_base64)
        write_data = write_data.replace("{{add_data}}", add_data)
        out.write(write_data)

    return


def make_static_html_ocr(pdf_path, pickle_path, html_path, template="template/template_vue.html", add_data=""):

    pickle_data = pickle.load(gzip.open(pickle_path, "rb"))
    ocr_result = pickle_data['ocr_result']
    translated_blocks = pickle_data['translated_blocks']

    out_dic = {'png_images':[], 'png_size':[], 'pages':[], 'pdf':'', 'paper_info': pickle_data['paper_info']}

    for idx, (page_idx, pdf_image, text_blocks_ocr, text_blocks) in enumerate(ocr_result):
        buffer = BytesIO()
        img_data = lp.draw_box(pdf_image, text_blocks_ocr, box_width=1, box_alpha=0.1, box_color='orange')
        img_data.save(buffer, 'png')
        img_str = base64.b64encode(buffer.getvalue()).decode("ascii")
        out_dic['png_images'].append(img_str)
        out_dic['png_size'].append({'height':img_data.height, 'width': img_data.width})

    for page_num, _ in enumerate(ocr_result):
        page_blocks = []
        for block in translated_blocks[page_num]:
            block_id = block['block_id']
            coords = block['box_info']
            block_data = {'block_id': int(block_id), 'texts': [], 'coords':coords }
            translated_block = block['translated']
            for translated in translated_block:
                block_data['texts'].append({
                    'best_is_norm': translated["best_is_norm"],
                    'en': html.escape(translated["en"]),
                    'ja_best': html.escape(translated["ja_best"]),
                    'ja_best_score': translated["ja_best_score"],
                    'ja_norm': html.escape(translated["ja_norm"]),
                    'ja_norm_score': translated["ja_norm_score"],          
                    'ja_parse': html.escape(translated["ja_parse"]),
                    'ja_parse_score': translated["ja_parse_score"]
                })
            page_blocks.append(block_data)
        out_dic['pages'].append(page_blocks)


    # with open(pdf_path, "rb") as in_pdf:
    #     out_dic['pdf'] = base64.b64encode(in_pdf.read()).decode("utf-8")
    

    with open(html_path, 'w') as out:
        with open(template) as in_html:
            out.write('{}'.format(in_html.read().replace('%%JSON_DATA%%', 'translated_data = {} ;'.format(json.dumps(out_dic)))))

    with gzip.open(pickle_path+'.json.gz', 'wt') as out:
        out.write('{}'.format(json.dumps(out_dic)))

    return


def main():
    parser = argparse.ArgumentParser(description='run fugu machine translator for pdf')
    parser.add_argument('config_file', help='config json file')
    parser.add_argument('--pdf', help='PDF file')
    parser.add_argument('--out', help='out pickle file')
    parser.add_argument('--mk_process', help='make marian process')
    parser.add_argument('--out_html', help='out html file')
    parser.add_argument('--ocr', help='use ocr mode')

    args = parser.parse_args()
    config = json.load(open(args.config_file))

    root_dir = config["app_root"]
    pickle_dir = config["pickle_dir"]
    pdf_file = ""
    if args.pdf:
        pdf_file = args.pdf
        out_pickle_file = args.out

    out_html_file = ""
    if args.out_html:
        out_html_file = args.out_html

    make_marian_conf = None
    if args.mk_process:
        make_marian_conf = config
    log_file = os.path.join(config["log_dir"], "pdf_server.log")

#    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, filename=log_file)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(0)

    def can_translate(en_txt):
        words = en_txt.split()
        if len(words) == 0:
            return False
        words_en = list(filter(lambda x: re.search("[a-zA-Z]", x), words))
        if len(words_en) == 0:
            return False
        if len(words_en) / len(words) < 0.2:
            return False
        if re.search("[a-zA-Z]", en_txt):
            return True
        else:
            return False
        
    fgmt = FuguJPNTranslator(config["pdf_ports"], retry_max=0, batch_size=5, use_constituency_parsing=False, can_translate_func=can_translate, use_sentence_tokenize='pysbd')

    if args.ocr:
        logger.info("pickle [{}] html [{}]".format(out_pickle_file, out_html_file))
        if not os.path.exists(out_pickle_file):
            logger.info("translate [{}] using ocr mode".format(pdf_file))
            ret =  pdf_translate_ocr(pdf_file, fgmt, make_marian_conf=make_marian_conf, logger=logger)
            title, title_ja, abstract = get_title_abstract(ret, fgmt, make_marian_conf=make_marian_conf, logger=logger)
            ret['paper_info'] = {'title':title, 'title_ja':title_ja, 'abstract':abstract}
            with gzip.open(out_pickle_file, 'wb') as out:
                pickle.dump(ret, out)
            logger.info(fgmt.get_and_clear_logs())
        else:
            logger.info("file {} exist. omit translating".format(out_pickle_file))

        make_static_html_ocr(pdf_file, out_pickle_file, out_html_file)
        #                 template=os.path.join(config["template_dir"], config["static_pdfhtml_template"]))        

        return 0

    if pdf_file:
        if not os.path.exists(out_pickle_file):
            logger.info("translate [{}]".format(pdf_file))
            ret = pdf_translate(pdf_file, fgmt, make_marian_conf=make_marian_conf, logger=logger)
            with open(out_pickle_file, "wb") as f:
                pickle.dump(ret, f)
            logger.info(fgmt.get_and_clear_logs())
        else:
            logger.info("file {} exist. omit translating".format(out_pickle_file))

    if out_html_file:
        logger.info("make html  [{}]".format(pdf_file))
        make_static_html(pdf_file, out_pickle_file, out_html_file,
                         template=os.path.join(config["template_dir"], config["static_pdfhtml_template"]))

    if not pdf_file:
        while True:
            logger.info("translate check db")
            ret = do_db(config["db_file"], config["pdf_dir"], config["pickle_dir"], fgmt,
                        make_marian_conf=make_marian_conf, logger=logger,
                        template=os.path.join(config["template_dir"], config["static_pdfhtml_template"]))
            if ret:
                logger.info("do_db[{}]".format(ret))
            logger.info("sleep {} sec".format(config["sleep_sec"]))
            time.sleep(config["sleep_sec"])


if __name__ == '__main__':
    main()





