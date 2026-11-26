"""원격 장비 측정 lock.

측정 명령은 장비에서 `flock -w <timeout> <lock_file> <명령>` 으로 실행한다 (RemoteClient._locked).
flock 은 프로세스가 끝나면(비정상 종료 포함) 자동으로 풀리므로 오래된 lock 정리가 필요 없다.
`dxconv remote check` 의 lock 칸에서 현재 사용 여부(free | busy)를 볼 수 있다.
"""
