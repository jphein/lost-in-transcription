# lost-in-transcription

Submission code for the Mozilla Data Collective / DrivenData
[Lost in Transcription](https://competitions.mozilladatacollective.com/competitions/group/lost-in-transcription/)
ASR challenge (Spanish-English, Indonesian-Javanese, Spanish-Nahuatl).

One `main.py` serves all three tracks. It routes each clip by the manifest's `language` column:
- Spanish/English go to faster-whisper large-v3.
- Javanese, Indonesian and Nahuatl go to Omnilingual ASR CTC (the 3B model by default).

The model weights are not in git (`models/` is ignored).

## Build and test a zip

```sh
# the omni zip (in-jv and sp-nh): the src/ code, es_words.txt, nah_vocab.tsv and models/omni-3b-v2-fp16.
# nah_vocab.tsv is built from the prepared Nahuatl dev set (scripts/prep_dev.py -> data/devrt/nahuatl).
scripts/pack_omni.sh                                    # -> dist/omni/submission.zip

# the Whisper zip (sp-en): src/*.py + models/faster-whisper-large-v3, stored (-0)
./pack.sh dist/sp-en/submission.zip faster-whisper-large-v3

# run it in the official runtime image, GPU on, network off
scripts/run_local.sh data/<dataset> dist/omni/submission.zip scratch/out.csv \
  -e LOST_IN_TRANSCRIPTION_IS_SMOKE=1
```

`pack.sh` copies only `src/*.py`, so it refuses omni model dirs. An omni zip without `es_words.txt` and
`nah_vocab.tsv` would still run, but its Nahuatl output would silently lose the Spanish lexicon.

`<dataset>` is a directory with `clips/` and `test_metadata.csv`
(`audio_filename,file_duration_seconds,language`), in the same layout as the competition's `/code_execution/data`.

## License

[MPL-2.0](LICENSE). The competition requires this license for winning solutions. External models: Whisper (MIT) and
Omnilingual ASR (Apache-2.0).

Data: `src/es_words.txt` is a list of 9,878 Spanish words derived from the FLEURS es_419 transcripts
(Conneau et al., 2022, Google; <https://huggingface.co/datasets/google/fleurs>), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes:
- the word types matching `[a-záéíóúüñ]+` were extracted from all three splits and deduplicated;
- 17 single-consonant tokens were removed;
- 25 common words that FLEURS lacks were added (e.g. ahorita, órale, okey, pesos).

`src/nah_vocab.tsv`, which `segment.py` reads, is built locally from the competition's dev set by
`scripts/build_nah_vocab.py`. It is not distributed in this repository.
