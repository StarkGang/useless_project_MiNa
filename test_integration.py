import requests
import time

base_url = 'http://127.0.0.1:8080'

# 1. Test Static Index
r = requests.get(base_url)
assert r.status_code == 200, f'Static index failed: {r.status_code}'
assert 'TheUnnecessaryFM' in r.text, 'Brand not found in HTML'
print('1. Static frontend served successfully!')

# 2. Test Analyze Endpoint
with open('/uploads/test_rain.wav', 'rb') as f:
    r = requests.post(f'{base_url}/api/analyze', files={'file': ('rain.wav', f, 'audio/wav')})
assert r.status_code == 200, f'Analyze failed: {r.text}'
res_analyze = r.json()
print('2. /api/analyze passed!')
print('   Classification:', res_analyze['classification'])
print('   Centroid:', res_analyze['dna']['spectral_centroid'], 'Hz')

# 3. Test Generate Endpoint
with open('/uploads/test_rain.wav', 'rb') as f:
    r = requests.post(
        f'{base_url}/api/generate',
        files={'file': ('rain.wav', f, 'audio/wav')},
        data={'beat_preference': 'minimal', 'energy_preference': 'balanced', 'seed': '777888999'}
    )
assert r.status_code == 200, f'Generate failed: {r.text}'
job_id = r.json()['job_id']
print(f'3. /api/generate submitted job: {job_id}')

# 4. Poll Status
for _ in range(60):
    time.sleep(1.5)
    status_r = requests.get(f'{base_url}/api/status/{job_id}')
    status_data = status_r.json()
    print('   Progress:', status_data['progress'], 'Stage:', status_data['stage'])
    if status_data['status'] == 'completed':
        break
    if status_data['status'] == 'failed':
        raise RuntimeError(f'Job failed: {status_data}')

# 5. Fetch Results
r = requests.get(f'{base_url}/api/result/{job_id}')
assert r.status_code == 200, f'Result failed: {r.text}'
res = r.json()
winner = res['winner']
print('4. /api/result retrieved!')
print('   Winner Scale:', winner['scale_name'], 'Tempo:', winner['tempo_bpm'], 'Score:', winner['score']['total'])
print('   Form:', winner['form'])
print('   Candidates count:', len(res['candidates']))

# 6. Stream Audio
r = requests.get(f'{base_url}{winner["audio_url"]}')
assert r.status_code == 200, f'Audio stream failed: {r.status_code}'
assert len(r.content) > 100000, 'Audio file too small'
print('5. Audio streamed successfully! Size:', len(r.content), 'bytes')
print('=== ALL API AND DSP INTEGRATION TESTS PASSED 100% ===')
