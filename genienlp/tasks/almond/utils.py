from tqdm import tqdm
import argparse
import json
import os


ISO_to_LANG = {'en': 'English', 'en-US': 'English', 'fa': 'Persian', 'it': 'Italian', 'zh': 'Chinese',
               'hr': 'Croatian', 'ja': 'Japanese', 'ko': 'Korean', 'ru': 'Russian', 'es': 'Spanish',
               'sv': 'Swedish', 'tr': 'Turkish', 'hi': 'Hindi', 'fr': 'French', 'de': 'German',
               'pl': 'Polsih', 'ar': 'Arabic', 'vi': 'Vietnamese', 'ji': 'Yiddish', 'pt': 'Portuguese',
               'el': 'Greek', 'he': 'Hebrew', 'si': 'Sinhala', 'ta': 'Tamil', 'fi': 'Finnish', 'cs': 'Czech',
               'no': 'Norwegian', 'tl': 'Filipino', 'da': 'Danish'}


CJK_RANGES = [
    (ord(u"\u3300"), ord(u"\u33ff")), (ord(u"\ufe30"), ord(u"\ufe4f")),   # compatibility ideographs
    (ord(u"\uf900"), ord(u"\ufaff")), (ord(u"\U0002F800"), ord(u"\U0002fa1f")),   # compatibility ideographs
    (ord(u'\u3040'), ord(u'\u309f')),   # Japanese Hiragana
    (ord(u"\u30a0"), ord(u"\u30ff")),   # Japanese Katakana
    (ord(u"\u2e80"), ord(u"\u2eff")),   # cjk radicals supplement
    (ord(u"\u4e00"), ord(u"\u9fff")),
    (ord(u"\u3400"), ord(u"\u4dbf")),
    (ord(u"\U00020000"), ord(u"\U0002a6df")),
    (ord(u"\U0002a700"), ord(u"\U0002b73f")),
    (ord(u"\U0002b740"), ord(u"\U0002b81f")),
    (ord(u"\U0002b820"), ord(u"\U0002ceaf"))
]

CJK_ADDONS = [ord(u"\u3001")]


def is_cjk_char(cp):
  return cp in CJK_ADDONS or any([range[0] <= cp <= range[1] for range in CJK_RANGES])

def is_entity(token):
    return token[0].isupper()

def is_device(token):
    return token[0] == '@'

def process_id(ex):
    id_ = ex.example_id.rsplit('/', 1)
    id_ = id_[0] if len(id_) == 1 else id_[1]
    # translated
    if id_[0] == 'T':
        id_ = id_[1:]
    return id_


def create_jsonl(path, examples, is_contextual):
    with open(path, 'w') as fout:
        for ex in examples:
            if is_contextual:
                fout.write(json.dumps({"sentence": ' '.join(ex.question)}) + '\n')
            else:
                fout.write(json.dumps({"sentence": ' '.join(ex.context)}) + '\n')

def bootleg_process(bootleg_input_dir, bootleg_model, examples, path, is_contextual, num_workers, logger):
    model = f'{bootleg_input_dir}/{bootleg_model}'

    cand_map = f'{bootleg_input_dir}/wiki_entity_data/entity_mappings/alias2qids_wiki.json'

    config_path = f'{model}/bootleg_config.json'
    
    # create jsonl file for examples
    jsonl_input_path = path.rsplit('.', 1)[0] + '.jsonl'
    jsonl_output_path = path.rsplit('.', 1)[0] + '_bootleg.jsonl'
    
    create_jsonl(jsonl_input_path, examples, is_contextual)
    
    # extract mentions
    from bootleg.extract_mentions import extract_mentions
    
    extract_mentions(in_filepath=jsonl_input_path, out_filepath=jsonl_output_path, cand_map_file=cand_map, logger=logger, num_workers=num_workers)
    
    data_dir = os.path.dirname(jsonl_output_path)
    data_file = os.path.basename(jsonl_output_path)
    
    entity_dir = f'{bootleg_input_dir}/wiki_entity_data'
    embed_dir = f'{bootleg_input_dir}/emb_data/'
    pretrained_bert = f'{bootleg_input_dir}/emb_data/pretrained_bert_models'
    
    mode = 'dump_preds'
    
    from bootleg.utils.parser_utils import get_full_config
    from bootleg.run import main

    overrides = ["--data_config.data_dir", data_dir,
                 "--data_config.entity_dir", entity_dir,
                 "--run_config.eval_batch_size", '30',
                 "--run_config.save_dir", 'results_temp',
                 "--run_config.init_checkpoint", model,
                 "--run_config.loglevel", 'debug',
                 "--data_config.test_dataset.file", data_file,
                 "--train_config.load_optimizer_from_ckpt", 'False',
                 "--data_config.emb_dir", embed_dir,
                 "--data_config.alias_cand_map", 'alias2qids_wiki.json',
                 "--data_config.word_embedding.cache_dir", pretrained_bert,
                 '--run_config.dataset_threads', '1',
                 '--run_config.dataloader_threads', '4']
    
    config_args = get_full_config(config_path, overrides)
    main(config_args, mode)


def chunk_file(input_src, chunk_files, chunk_size, num_chunks):
    chunk_id = 0
    num_lines_in_chunk = 0
    all_out_files = [open(chunk_files[chunk_id], 'w') for chunk_id in range(num_chunks)]
    with open(input_src, 'r', encoding='utf-8') as in_file:
        for line in in_file:
            all_out_files[chunk_id].write(line)
            num_lines_in_chunk += 1
            if num_lines_in_chunk == chunk_size:
                chunk_id += 1
                num_lines_in_chunk = 0
                if chunk_id == num_chunks:
                    break

    for file in all_out_files:
        file.close()


def process(args):
    path = args['in_file']
    chunk_size = args['chunk_size']
    dir_name = args['dir_name']
    example_batch_size = args['example_batch_size']
    make_process_example = args['make_process_example']
    kwargs = args['kwargs']
    
    chunk_examples = []
    
    batch = []
    last_batch = False
    for i, line in tqdm(enumerate(open(path, 'r', encoding='utf-8')), total=chunk_size):
        parts = line.strip().split('\t')
        batch.append(parts)
        if len(chunk_examples) + example_batch_size > chunk_size:
            # trim batch
            batch = batch[:chunk_size - len(chunk_examples)]
            last_batch = True
        if len(batch) % example_batch_size != 0 and not last_batch:
            continue
        assert len(batch) == 1
        batch = batch[0]
        chunk_examples.append(make_process_example(batch, dir_name, **kwargs))
        batch = []
        if len(chunk_examples) >= chunk_size:
            break
    
    return chunk_examples




