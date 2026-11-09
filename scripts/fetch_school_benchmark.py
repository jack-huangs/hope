"""Fetch only non-secret benchmark records from the authorized campus host."""
import getpass
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / '.qwen-deploy-tools'))
import paramiko

client = paramiko.SSHClient()
client.load_system_host_keys(str(pathlib.Path.home() / '.ssh/known_hosts'))
client.connect('10.108.12.110', username='hjj', password=getpass.getpass('SSH password: '),
               look_for_keys=False, allow_agent=False, timeout=15)
out = pathlib.Path('runs/school_api_benchmark_20261108_201330')
out.mkdir(parents=True, exist_ok=True)
with client.open_sftp() as sftp:
    for name in ['results.json', 'summary.json', 'long_result.json']:
        sftp.get('/home/hjj/school-api-benchmark/' + out.as_posix() + '/' + name, str(out / name))
        print('Fetched', out / name, flush=True)
client.close()
