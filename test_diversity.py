import requests
import time

base_url = 'http://127.0.0.1:8080'

for name, filepath in [('footsteps', 'useless_project_MiNa/uploads/test_footsteps.wav'), ('fan', 'useless_project_Mina/uploads/test_fan.wav')]:
    print(f'Testing {name}...')
    with open(filepath, 'rb') as f:
        r = requests.post(f'{base_url}/api/generate', files={'file': (f'{name}.wav', f, 'audio/wav')})
    assert r.status_code == 200, r.text
    job_id = r.json()['job_id']
    for _ in range(60):
        time.sleep(2.0)
        s = requests.get(f'{base_url}/api/status/{job_id}').json()
        if s['status'] == 'completed':
            break
        if s['status'] == 'failed':
            raise RuntimeError(f'{name} failed: {s}')

    res = requests.get(f'{base_url}/api/result/{job_id}').json()
    winner = res['winner']
    print(f'  [OK] {name} completed successfully!')
    print(f'    Duration: {winner["duration"]}s')
    print(f'    Tempo: {winner["tempo_bpm"]} BPM')
    print(f'    Scale: {winner["scale_name"]}')
    print(f'    Form: {winner["form"]}')
    print(f'    Source Ratio: {winner["score"]["source_usage_ratio"] * 100}%')
    print(f'    Synthetic Ratio: {winner["score"]["synthetic_audio_ratio"] * 100}%')
    print(f'    Palette Slices: {winner["palette"]["total_slices"]}')
    print(f'    Score: {winner["score"]["total"]}')
    print(f'    Stems: {list(winner["stems"].keys())}')

print('\n=== ALL DIVERSITY SOUND TESTS COMPLETED SUCCESSFULLY! ===')
