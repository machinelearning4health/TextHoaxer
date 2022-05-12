# Please adjust your path according to your own environment

python3  TextHoaxer_classification.py  --dataset_path data/mr --target_model wordCNN --target_dataset mr --target_model_path cnn/mr         --USE_cache_path  original/attackdownload --max_seq_length 256         --sim_score_window 40         --nclasses 2  --counter_fitting_cos_sim_path mat.txt --word_embeddings_path  glove.6B.200d.txt --budget 1000

python3  TextHoaxer_classification.py --dataset_path data/ag --target_model wordCNN --target_dataset ag --target_model_path  cnn/ag         --USE_cache_path  original/attackdownload --max_seq_length 256         --sim_score_window 40         --nclasses 4  --counter_fitting_cos_sim_path mat.txt --word_embeddings_path  glove.6B.200d.txt  --budget 1000

python3  TextHoaxer_classification.py --dataset_path data/yahoo --target_model wordCNN   --target_dataset yahoo   --target_model_path  cnn/yahoo         --USE_cache_path  original/attackdownload         --max_seq_length 256         --sim_score_window 40         --nclasses 10  --counter_fitting_cos_sim_path mat.txt  --word_embeddings_path  glove.6B.200d.txt --budget 1000

python3  TextHoaxer_classification.py --dataset_path data/yelp  --target_model wordCNN   --target_dataset yelp         --target_model_path  cnn/yelp        --USE_cache_path  original/attackdownload         --max_seq_length 256         --sim_score_window 40         --nclasses 2  --counter_fitting_cos_sim_path mat.txt --word_embeddings_path  glove.6B.200d.txt --budget 1000

python3  TextHoaxer_classification.py --dataset_path data/imdb          --target_model wordCNN   --target_dataset imdb         --target_model_path  cnn/imdb         --USE_cache_path  original/attackdownload         --max_seq_length 256         --sim_score_window 40         --nclasses 2  --counter_fitting_cos_sim_path mat.txt --word_embeddings_path  glove.6B.200d.txt --budget 1000

python3  TextHoaxer_nli.py  --dataset_path data/mnli_matched  --target_model bert   --target_dataset mnli_matched      --target_model_path bert/mnli  --USE_cache_path  original/nli_cache  --sim_score_window 40   --counter_fitting_cos_sim_path mat.txt --output_dir results/mnli_matched_bert_ours  --budget 1000

python3  TextHoaxer_nli.py  --dataset_path data/mnli_mismatched  --target_model bert   --target_dataset mnli_mismatched      --target_model_path  bert/mnli  --USE_cache_path  original/nli_cache  --sim_score_window 40   --counter_fitting_cos_sim_path mat.txt --output_dir results/mnli_mismatched_bert_ours  --budget 1000

python3  TextHoaxer_nli.py  --dataset_path data/snli  --target_model bert   --target_dataset snli         --target_model_path  bert/snli  --USE_cache_path original/nli_cache  --sim_score_window 40   --counter_fitting_cos_sim_path mat.txt --output_dir results/snli_bert_ours --budget 1000