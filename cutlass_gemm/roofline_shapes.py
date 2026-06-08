PEAK_FP4 = 2000e12
BW = 1792e9
shapes = [
    (16384, 2048, 2048,  "1 q/o fprop+dgrad"),
    (16384, 1024, 2048,  "2 k/v fprop"),
    (16384, 2048, 1024,  "3 k/v dgrad"),
    (16384, 8192, 2048,  "4 up fprop"),
    (16384, 2048, 8192,  "5 down fprop"),
    (2048,  2048, 16384, "6 q/o wgrad"),
    (1024,  2048, 16384, "7 k/v wgrad"),
    (8192,  2048, 16384, "8 up wgrad"),
    (2048,  8192, 16384, "9 down wgrad"),
]
ridge = PEAK_FP4 / BW
print(f"Ridge: {ridge:.1f} FLOP/byte")
print(f"{'name':<20}{'M':>6}{'N':>6}{'K':>7}{'AI':>8}{'bound':>6}{'ceil_TF':>9}{'85%ceil':>9}{'tiles128':>9}{'waves':>7}")
for M,N,K,who in shapes:
    flop=2*M*N*K
    byts=M*K*0.5+M*K/16 + K*N*0.5+K*N/16 + M*N*2
    ai=flop/byts
    bound="comp" if ai>=ridge else "BW"
    ceil=PEAK_FP4 if bound=="comp" else ai*BW
    tiles=((M+127)//128)*((N+127)//128)
    waves=tiles/188
    print(f"{who:<20}{M:>6}{N:>6}{K:>7}{ai:>8.1f}{bound:>6}{ceil/1e12:>9.0f}{0.85*ceil/1e12:>9.0f}{tiles:>9}{waves:>7.2f}")
