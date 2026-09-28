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
# pack src/*.py + chosen model dirs (from models/) into a stored (-0) zip
./pack.sh dist/in-jv/submission.zip omni-3b-v2-fp16

# run it in the official runtime image, GPU on, network off
scripts/run_local.sh data/<dataset> dist/in-jv/submission.zip scratch/out.csv \
  -e LOST_IN_TRANSCRIPTION_IS_SMOKE=1
```

`<dataset>` is a directory with `clips/` and `test_metadata.csv`
(`audio_filename,file_duration_seconds,language`), in the same layout as the competition's `/code_execution/data`.

## License

[MPL-2.0](LICENSE). The competition requires this license for winning solutions. External models: Whisper (MIT) and
Omnilingual ASR (Apache-2.0).
