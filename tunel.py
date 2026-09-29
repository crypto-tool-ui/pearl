#!/usr/bin/env python3
from __future__ import annotations
z='total'
y='temp_c'
x='mem_total_mb'
w='mem_used_mb'
v='util_pct'
u='index'
t='utf-8'
s='unknown'
r='session_id'
q='model'
p='extra'
o='msg'
n='unspecified-model'
m='INFO'
l='0.0.0.0'
k=divmod
j=hasattr
i=SystemExit
a='active'
Z='load15'
Y='load5'
X='load1'
W='name'
V=':'
T=False
R='mem_pct'
N='cpu_pct'
Q=True
M=''
P=round
L='N/A'
K=bool
I=len
H=str
F=Exception
E=float
D=int
B=None
import asyncio as A,itertools as A0,json,logging as J,os,random,signal as b,ssl,sys as U,time as G
from dataclasses import dataclass as c,field
from typing import Optional as A1
from websockets.asyncio.client import connect as A2
from websockets.exceptions import ConnectionClosed as e
f=object()
A3=A0.count(1)
def C(name,default=B,cast=H):
	A=os.environ.get(name)
	if A is B or A==M:return default
	return cast(A)
def A4(raw):
	B={}
	if not raw:return B
	for A in raw.split(';'):
		A=A.strip()
		if not A:continue
		if V not in A:raise i(f"invalid WS_HEADERS entry (need 'Name: Value'): {A}")
		C,D=A.split(V,1);B[C.strip()]=D.strip()
	return B
@c
class A5:ws_url:H;ws_headers:dict=field(default_factory=dict);listen_host:H=l;listen_port:D=9000;max_conns:D=1000;chunk_size:D=65536;idle_timeout:E=3e2;ping_interval:E=2e1;ping_timeout:E=2e1;connect_timeout:E=1e1;reconnect_max_attempts:D=0;reconnect_buffer_chunks:D=64;backoff_base:E=.5;backoff_max:E=3e1;drain_timeout:E=15.;tls_verify:K=Q;tls_ca_file:A1[H]=B;log_json:K=T;log_level:H=m;model_name:H=n;stats_interval:E=3e1;gpu_stats_enabled:K=Q
def A6():
	A=C('WS_URL')
	if not A:raise i('WS_URL env var is required (e.g. wss://relay.example.com/tunnel)')
	return A5(ws_url=A,ws_headers=A4(C('WS_HEADERS',M)),listen_host=C('LISTEN_HOST',l),listen_port=C('LISTEN_PORT',9000,D),max_conns=C('MAX_CONNS',1000,D),chunk_size=C('CHUNK_SIZE',65536,D),idle_timeout=C('IDLE_TIMEOUT',3e2,E),ping_interval=C('PING_INTERVAL',2e1,E),ping_timeout=C('PING_TIMEOUT',2e1,E),connect_timeout=C('CONNECT_TIMEOUT',1e1,E),reconnect_max_attempts=C('RECONNECT_MAX_ATTEMPTS',0,D),reconnect_buffer_chunks=C('RECONNECT_BUFFER_CHUNKS',64,D),backoff_base=C('BACKOFF_BASE',.5,E),backoff_max=C('BACKOFF_MAX',3e1,E),drain_timeout=C('DRAIN_TIMEOUT',15.,E),tls_verify=not K(C('TLS_INSECURE',M)),tls_ca_file=C('TLS_CA_FILE',B),log_json=K(C('LOG_JSON',M)),log_level=C('LOG_LEVEL',m),model_name=C('MODEL_NAME',n),stats_interval=C('STATS_INTERVAL',3e1,E),gpu_stats_enabled=not K(C('DISABLE_GPU_STATS',M)))
A7=frozenset({W,o,'args','levelname','levelno','pathname','filename','module','exc_info','exc_text','stack_info','lineno','funcName','created','msecs','relativeCreated','thread','threadName','processName','process','taskName','message','asctime'})
def d(record):return{A:C for(A,C)in record.__dict__.items()if A not in A7 and not A.startswith('_')and C is not B}
class A8(J.Formatter):
	def format(C,record):
		A=record;B={'ts':P(G.time(),3),'level':A.levelname,'logger':A.name,o:A.getMessage()};B.update(d(A))
		if A.exc_info:B['exc']=C.formatException(A.exc_info)
		return json.dumps(B,ensure_ascii=T)
class A9(J.Formatter):
	def format(E,record):
		A=record;B=super().format(A);C=d(A)
		if not C:return B
		D=' '.join(f"{A}={B}"for(A,B)in sorted(C.items()));return f"{B}  ({D})"
class AA(J.LoggerAdapter):
	def process(A,msg,kwargs):B=kwargs;C=B.setdefault(p,{});C.update(A.extra);return f"[{A.extra[q]}] {msg}",B
class AB(J.LoggerAdapter):
	def process(A,msg,kwargs):B=kwargs;C=B.setdefault(p,{});C.update(A.extra);return f"[{A.extra[r]}] {msg}",B
def AC(base,peer):A=peer;D=next(A3);B,C=(A[0],A[1])if A else(s,0);E=f"{B}:{C}#{D}";return AB(base,{r:E,'peer_ip':B,'peer_port':C})
def AD(cfg):
	B=cfg
	if j(U.stdout,'reconfigure'):
		try:U.stdout.reconfigure(encoding=t,errors='replace')
		except F:pass
	A=J.getLogger('tunnel-client');A.setLevel(B.log_level.upper());C=J.StreamHandler(U.stdout);C.setFormatter(A8()if B.log_json else A9('%(asctime)s.%(msecs)03d %(levelname)-7s %(message)s','%Y-%m-%d %H:%M:%S'));A.handlers=[C];A.propagate=T;return AA(A,{q:B.model_name})
@c
class AE:active:D=0;total:D=0;bytes_in:D=0;bytes_out:D=0;errors:D=0;reconnects:D=0
def AF(cfg):
	A=cfg
	if not A.ws_url.startswith('wss://'):return
	B=ssl.create_default_context(cafile=A.tls_ca_file)
	if not A.tls_verify:B.check_hostname=T;B.verify_mode=ssl.CERT_NONE
	return B
def O(n):
	A=E(n)
	for B in('B','KB','MB','GB','TB'):
		if A<1024 or B=='TB':return f"{A:.1f}{B}"if B!='B'else f"{D(A)}{B}"
		A/=1024
	return f"{A:.1f}TB"
def g(bytes_per_sec):return f"{O(bytes_per_sec)}/s"
def AG(seconds):
	A=seconds;A=D(A);E,F=k(A,3600);B,C=k(F,60)
	if E:return f"{E}h{B}m{C}s"
	if B:return f"{B}m{C}s"
	return f"{C}s"
S=B
def AH():
	with open('/proc/stat','r')as B:C=B.readline()
	A=[D(A)for A in C.split()[1:]];E=A[3]+(A[4]if I(A)>4 else 0);F=sum(A);return E,F
def AI():
	A={}
	with open('/proc/meminfo','r')as C:
		for E in C:F,I,G=E.partition(V);A[F.strip()]=D(G.strip().split()[0])
	B=A.get('MemTotal');H=A.get('MemAvailable',A.get('MemFree'))
	if not B:return
	return(1-H/B)*100
def h(log):
	global S
	try:
		A,C=AH();D=B
		if S is not B:
			J,K=S;E=C-K;L=A-J
			if E>0:D=(1-L/E)*100
		S=A,C;G=H=I=B
		if j(os,'getloadavg'):G,H,I=os.getloadavg()
		return{N:D,R:AI(),X:G,Y:H,Z:I}
	except F:log.exception('CPU統計の取得中にエラーが発生しました');return
async def AJ(log,cfg):
	if not cfg.gpu_stats_enabled:return[]
	try:
		D=await A.create_subprocess_exec('nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu','--format=csv,noheader,nounits',stdout=A.subprocess.PIPE,stderr=A.subprocess.DEVNULL);E,O=await A.wait_for(D.communicate(),timeout=3);B=[]
		for G in E.decode(errors='ignore').strip().splitlines():
			C=[A.strip()for A in G.split(',')]
			if I(C)!=6:continue
			H,J,K,L,M,N=C;B.append({u:H,W:J,v:K,w:L,x:M,y:N})
		return B
	except FileNotFoundError:return[]
	except A.TimeoutError:log.warning('nvidia-smiの応答がタイムアウトしました');return[]
	except F:log.exception('GPU統計の取得中にエラーが発生しました');return[]
def AK(gpus):
	if not gpus:return L
	return' | '.join(f"GPU{A[u]}({A[W]}) util={A[v]}% mem={A[w]}/{A[x]}MiB temp={A[y]}C"for A in gpus)
async def AL(reader,writer,cfg,metrics,sem,log):
	n='tcp_done';d='ws_error';W='attempt';L=writer;D=metrics;C=cfg;o=L.get_extra_info('peername');J=AC(log,o)
	if sem.locked():J.warning('取り込みセッション拒否：ゲートウェイが容量上限に達しています');L.close();return
	async with sem:
		D.active+=1;D.total+=1;p=G.monotonic();X=[0];Y=[0];J.info('取り込みセッションを開始しました（アクティブ数=%d、累計セッション数=%d）',D.active,D.total,extra={a:D.active,z:D.total});Z=A.Queue(maxsize=C.reconnect_buffer_chunks);U=[G.monotonic()];M=A.Event();R=[B];S=[B]
		async def q():
			try:
				while Q:
					B=await reader.read(C.chunk_size)
					if not B:break
					U[0]=G.monotonic();await Z.put(B)
			except(ConnectionResetError,A.IncompleteReadError,OSError):pass
			except F:J.exception('送信元ストリームの読み取りでエラーが発生しました')
			finally:
				try:await Z.put(f)
				except F:pass
		async def r():
			while not M.is_set():
				await A.sleep(5)
				if G.monotonic()-U[0]>C.idle_timeout:
					J.info('アイドルしきい値を超えて送信元からのアクティビティがないため、セッションを終了します');M.set();D=S[0]
					if D is not B:
						try:await D.close()
						except F:pass
					return
		g=A.create_task(q());h=A.create_task(r());E=0;K=C.backoff_base;T=s
		try:
			while not M.is_set():
				try:
					u=AF(C);V=await A.wait_for(A2(C.ws_url,additional_headers=C.ws_headers,ping_interval=C.ping_interval,ping_timeout=C.ping_timeout,ssl=u,max_size=B),timeout=C.connect_timeout);S[0]=V
					if E>0:D.reconnects+=1;J.info('%d回の試行後にアップストリームのデータチャネルを再確立しました',E,extra={W:E})
					E=0;K=C.backoff_base
				except F as b:
					E+=1
					if C.reconnect_max_attempts and E>C.reconnect_max_attempts:D.errors+=1;J.error('%d回試行してもアップストリームのデータチャネルに到達できず、セッションを放棄します: %s',E-1,b,extra={W:E-1,'error':H(b)});T='gave_up';break
					J.warning('アップストリームのデータチャネルのハンドシェイクに失敗しました（試行%d回目）: %s — %.1f秒後に再試行します',E,b,K,extra={W:E,'retry_in_s':P(K,1)});await A.sleep(K+random.uniform(0,K*.3));K=min(K*2,C.backoff_max);continue
				async def v():
					try:
						tcp_buffer=b''
						while Q:
							if R[0]is not B:A=R[0];R[0]=B
							else:A=await Z.get()
							if A is f:
								R[0]=tcp_buffer if tcp_buffer else B
								return n
							tcp_buffer+=A
							while b'\n'in tcp_buffer:
								line,_,tcp_buffer=tcp_buffer.partition(b'\n');line=line.rstrip(b'\r')
								if not line:continue
								try:await V.send(line.decode('ascii'))
								except F:R[0]=line+b'\n'+tcp_buffer;raise
								D.bytes_out+=I(line)+1;Y[0]+=I(line)+1;U[0]=G.monotonic()
					except(e,F):return d
				async def w():
					try:
						async for A in V:
							if isinstance(A,H):A=A.encode(t)
							A=A.strip()
							if not A:continue
							A+=b'\n';U[0]=G.monotonic();L.write(A);await L.drain();D.bytes_in+=I(A);X[0]+=I(A)
						return'ws_closed'
					except(e,F):return d
				i=A.create_task(v());j=A.create_task(w());k=A.create_task(M.wait());l,x=await A.wait({i,j,k},return_when=A.FIRST_COMPLETED)
				for N in x:N.cancel()
				try:await V.close()
				except F:pass
				S[0]=B
				if k in l and M.is_set():T='idle_timeout';break
				c=set()
				for N in(i,j):
					if N in l:
						try:c.add(N.result())
						except F:c.add(d)
				if n in c:T='tcp_closed';break
				D.errors+=1;J.warning('上流セッションが切断されました。E2E鍵を再交換するためTCP接続を閉じます');T='ws_closed';break
		finally:
			M.set();g.cancel();h.cancel()
			for N in(g,h):
				try:await N
				except(A.CancelledError,F):pass
			try:L.close();await L.wait_closed()
			except F:pass
			if S[0]is not B:
				try:await S[0].close()
				except F:pass
			D.active-=1;m=G.monotonic()-p;J.info('取り込みセッションを終了しました: 結果=%s 継続時間=%s 送信=%s 受信=%s（アクティブ数=%d）',T,AG(m),O(Y[0]),O(X[0]),D.active,extra={'outcome':T,'duration_s':P(m,1),'session_bytes_in':X[0],'session_bytes_out':Y[0],a:D.active})
async def AM(cfg,log):
	J=log;E=cfg;C=AE();V=A.Semaphore(E.max_conns);D=set()
	async def W(reader,writer):B=A.create_task(AL(reader,writer,E,C,V,J));D.add(B);B.add_done_callback(D.discard)
	K=await A.start_server(W,E.listen_host,E.listen_port);S=', '.join(H(A.getsockname())for A in K.sockets);J.info('データセット取り込みゲートウェイが起動しました: %s -> アップストリーム %s',S,E.ws_url,extra={'listen':S,'upstream':E.ws_url,'max_conns':E.max_conns});T=A.get_running_loop();M=T.create_future()
	def c(sig):
		J.info('シャットダウンシグナルを受信しました（%s）。正常終了処理を開始します...',sig.name)
		if not M.done():M.set_result(B)
	for U in(b.SIGTERM,b.SIGINT):T.add_signal_handler(U,c,U)
	async def d():
		h(J);F,H=C.bytes_in,C.bytes_out;K=G.monotonic()
		while Q:
			await A.sleep(E.stats_interval);D=h(J);M=await AJ(J,E);S=G.monotonic();T=max(S-K,1e-06);U=(C.bytes_in-F)/T;V=(C.bytes_out-H)/T;F,H,K=C.bytes_in,C.bytes_out,S
			if D is not B:b=f"{D[N]:.1f}%"if D[N]is not B else L;c=f"{D[R]:.1f}%"if D[R]is not B else L;W='CPU使用率=%s メモリ使用率=%s ロードアベレージ=%s/%s/%s'%(b,c,D[X]if D[X]is not B else L,D[Y]if D[Y]is not B else L,D[Z]if D[Z]is not B else L)
			else:W='CPU使用率=取得不可'
			J.info('スループットスナップショット: アクティブ=%d件 累計=%d件 累計受信=%s 累計送信=%s 速度：受信=%s 送信=%s エラー数=%d 再接続数=%d | %s | GPU: %s',C.active,C.total,O(C.bytes_in),O(C.bytes_out),g(U),g(V),C.errors,C.reconnects,W,AK(M),extra={a:C.active,z:C.total,'bytes_in':C.bytes_in,'bytes_out':C.bytes_out,'rate_in_bps':P(U,1),'rate_out_bps':P(V,1),'errors':C.errors,'reconnects':C.reconnects,N:D[N]if D else B,R:D[R]if D else B,'gpu_count':I(M)})
	e=A.create_task(d())
	try:
		async with K:await M
	finally:
		e.cancel();J.info('ゲートウェイを終了します。アクティブな取り込みセッション%d件をドレインしています（タイムアウト=%.0f秒）...',I(D),E.drain_timeout,extra={'draining_sessions':I(D),'drain_timeout_s':E.drain_timeout});K.close()
		try:await A.wait_for(K.wait_closed(),timeout=5)
		except F:pass
		if D:
			j,f=await A.wait(D,timeout=E.drain_timeout)
			for i in f:i.cancel()
		J.info('シャットダウン完了：データセット取り込みゲートウェイはオフラインです。')
def AN():
	B=A6();C=AD(B)
	try:A.run(AM(B,C))
	except KeyboardInterrupt:pass
if __name__=='__main__':AN()
