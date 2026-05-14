
/home/user/vortex/target/release/deps/funnel_patterns-b50fc531d4f952e4:     file format elf64-x86-64


Disassembly of section .text:

0000000000059520 <_ZN15funnel_patterns19pat_chunked_aligned17h862079dcf94e24a3E>:
   59520:	lea    0x8(%rdi),%rax
   59524:	vpbroadcastq %rsi,%zmm0
   5952a:	xor    %ecx,%ecx
   5952c:	vpbroadcastq -0x3a836(%rip),%zmm4        # 1ed00 <anon.576c71cbdbd6e3deefe97f3cb576ad67.20.llvm.9236128649512152758+0x1dc>
   59536:	vpbroadcastq -0x3a7e8(%rip),%zmm5        # 1ed58 <anon.576c71cbdbd6e3deefe97f3cb576ad67.20.llvm.9236128649512152758+0x234>
   59540:	vmovdqa64 -0x39bca(%rip),%zmm6        # 1f980 <anon.3c8f25b610820e10fad07aea247a682a.12.llvm.15771806882712607681+0xb68>
   5954a:	vpbroadcastq -0x3a76c(%rip),%zmm7        # 1ede8 <anon.576c71cbdbd6e3deefe97f3cb576ad67.20.llvm.9236128649512152758+0x2c4>
   59554:	vmovdqa64 -0x39b9e(%rip),%zmm8        # 1f9c0 <anon.3c8f25b610820e10fad07aea247a682a.12.llvm.15771806882712607681+0xba8>
   5955e:	vmovdqa64 -0x39b68(%rip),%zmm9        # 1fa00 <anon.3c8f25b610820e10fad07aea247a682a.12.llvm.15771806882712607681+0xbe8>
   59568:	vmovdqa64 -0x39b32(%rip),%zmm10        # 1fa40 <anon.3c8f25b610820e10fad07aea247a682a.12.llvm.15771806882712607681+0xc28>
   59572:	vmovdqa64 -0x39afc(%rip),%zmm11        # 1fa80 <anon.3c8f25b610820e10fad07aea247a682a.12.llvm.15771806882712607681+0xc68>
   5957c:	vmovdqa64 -0x39ac6(%rip),%zmm12        # 1fac0 <anon.3c8f25b610820e10fad07aea247a682a.12.llvm.15771806882712607681+0xca8>
   59586:	vmovdqa64 -0x39a90(%rip),%zmm13        # 1fb00 <anon.3c8f25b610820e10fad07aea247a682a.12.llvm.15771806882712607681+0xce8>
   59590:	vmovdqa64 -0x39a5a(%rip),%zmm14        # 1fb40 <anon.3c8f25b610820e10fad07aea247a682a.12.llvm.15771806882712607681+0xd28>
   5959a:	vmovdqa64 -0x39a24(%rip),%zmm15        # 1fb80 <anon.3c8f25b610820e10fad07aea247a682a.12.llvm.15771806882712607681+0xd68>
   595a4:	vmovdqa64 -0x399ee(%rip),%zmm16        # 1fbc0 <anon.3c8f25b610820e10fad07aea247a682a.12.llvm.15771806882712607681+0xda8>
   595ae:	vmovdqa64 -0x399b8(%rip),%zmm17        # 1fc00 <anon.3c8f25b610820e10fad07aea247a682a.12.llvm.15771806882712607681+0xde8>
   595b8:	vmovdqa64 -0x39982(%rip),%zmm18        # 1fc40 <anon.3c8f25b610820e10fad07aea247a682a.12.llvm.15771806882712607681+0xe28>
   595c2:	vmovdqa64 -0x3994c(%rip),%zmm19        # 1fc80 <anon.3c8f25b610820e10fad07aea247a682a.12.llvm.15771806882712607681+0xe68>
   595cc:	vmovdqa64 -0x39916(%rip),%zmm20        # 1fcc0 <anon.3c8f25b610820e10fad07aea247a682a.12.llvm.15771806882712607681+0xea8>
   595d6:	vmovdqa64 -0x398e0(%rip),%zmm21        # 1fd00 <anon.3c8f25b610820e10fad07aea247a682a.12.llvm.15771806882712607681+0xee8>
   595e0:	vmovdqa64 -0x398aa(%rip),%zmm22        # 1fd40 <anon.3c8f25b610820e10fad07aea247a682a.12.llvm.15771806882712607681+0xf28>
   595ea:	vmovdqa64 -0x39874(%rip),%zmm23        # 1fd80 <anon.3c8f25b610820e10fad07aea247a682a.12.llvm.15771806882712607681+0xf68>
   595f4:	vmovdqa64 -0x3983e(%rip),%zmm24        # 1fdc0 <anon.3c8f25b610820e10fad07aea247a682a.12.llvm.15771806882712607681+0xfa8>
   595fe:	vmovdqa64 -0x39808(%rip),%zmm25        # 1fe00 <anon.3c8f25b610820e10fad07aea247a682a.12.llvm.15771806882712607681+0xfe8>
   59608:	vmovdqa64 -0x397d2(%rip),%zmm26        # 1fe40 <anon.3c8f25b610820e10fad07aea247a682a.12.llvm.15771806882712607681+0x1028>
   59612:	vmovdqa64 -0x3979c(%rip),%zmm27        # 1fe80 <anon.3c8f25b610820e10fad07aea247a682a.12.llvm.15771806882712607681+0x1068>
   5961c:	vmovdqa64 -0x39766(%rip),%zmm28        # 1fec0 <anon.3c8f25b610820e10fad07aea247a682a.12.llvm.15771806882712607681+0x10a8>
   59626:	cs nopw 0x0(%rax,%rax,1)
   59630:	vpbroadcastq %rcx,%zmm29
   59636:	vpmullq -0x3a8f0(%rip){1to8},%zmm29,%zmm30        # 1ed50 <anon.576c71cbdbd6e3deefe97f3cb576ad67.20.llvm.9236128649512152758+0x22c>
   59640:	vpaddq -0x39d0a(%rip),%zmm30,%zmm31        # 1f940 <anon.3c8f25b610820e10fad07aea247a682a.12.llvm.15771806882712607681+0xb28>
   5964a:	vpsrlq $0x6,%zmm31,%zmm1
   59651:	vpminuq %zmm4,%zmm1,%zmm2
   59657:	kxnorw %k0,%k0,%k1
   5965b:	vpxor  %xmm3,%xmm3,%xmm3
   5965f:	kxnorw %k0,%k0,%k2
   59663:	vpgatherqq (%rdi,%zmm1,8),%zmm3{%k1}
   5966a:	vpxor  %xmm1,%xmm1,%xmm1
   5966e:	vpgatherqq (%rax,%zmm2,8),%zmm1{%k2}
   59675:	vpandq %zmm5,%zmm31,%zmm2
   5967b:	vpsrlvq %zmm2,%zmm3,%zmm2
   59681:	vpsubq %zmm30,%zmm6,%zmm3
   59687:	vpaddq -0x39d91(%rip),%zmm29,%zmm31        # 1f900 <anon.3c8f25b610820e10fad07aea247a682a.12.llvm.15771806882712607681+0xae8>
   59691:	vpandq %zmm5,%zmm3,%zmm3
   59697:	vpsllvq %zmm3,%zmm1,%zmm1
   5969d:	vpternlogq $0xc8,%zmm2,%zmm7,%zmm1
   596a4:	vpaddq %zmm0,%zmm1,%zmm1
   596aa:	kxnorw %k0,%k0,%k1
   596ae:	vpscatterqq %zmm1,(%rdx,%zmm31,8){%k1}
   596b5:	vpaddq %zmm9,%zmm30,%zmm1
   596bb:	vpsrlq $0x6,%zmm1,%zmm2
   596c2:	vpminuq %zmm4,%zmm2,%zmm3
   596c8:	kxnorw %k0,%k0,%k1
   596cc:	vpxord %xmm31,%xmm31,%xmm31
   596d2:	kxnorw %k0,%k0,%k2
   596d6:	vpgatherqq (%rdi,%zmm2,8),%zmm31{%k1}
   596dd:	vpxor  %xmm2,%xmm2,%xmm2
   596e1:	vpgatherqq (%rax,%zmm3,8),%zmm2{%k2}
   596e8:	vpandq %zmm5,%zmm1,%zmm1
   596ee:	vpsrlvq %zmm1,%zmm31,%zmm1
   596f4:	vpsubq %zmm30,%zmm10,%zmm3
   596fa:	vpaddq %zmm8,%zmm29,%zmm31
   59700:	vpandq %zmm5,%zmm3,%zmm3
   59706:	vpsllvq %zmm3,%zmm2,%zmm2
   5970c:	vpternlogq $0xc8,%zmm1,%zmm7,%zmm2
   59713:	vpaddq %zmm0,%zmm2,%zmm1
   59719:	kxnorw %k0,%k0,%k1
   5971d:	vpscatterqq %zmm1,(%rdx,%zmm31,8){%k1}
   59724:	vpaddq %zmm12,%zmm30,%zmm1
   5972a:	vpsrlq $0x6,%zmm1,%zmm2
   59731:	vpminuq %zmm4,%zmm2,%zmm3
   59737:	kxnorw %k0,%k0,%k1
   5973b:	vpxord %xmm31,%xmm31,%xmm31
   59741:	kxnorw %k0,%k0,%k2
   59745:	vpgatherqq (%rdi,%zmm2,8),%zmm31{%k1}
   5974c:	vpxor  %xmm2,%xmm2,%xmm2
   59750:	vpgatherqq (%rax,%zmm3,8),%zmm2{%k2}
   59757:	vpandq %zmm5,%zmm1,%zmm1
   5975d:	vpsrlvq %zmm1,%zmm31,%zmm1
   59763:	vpsubq %zmm30,%zmm13,%zmm3
   59769:	vpaddq %zmm11,%zmm29,%zmm31
   5976f:	vpandq %zmm5,%zmm3,%zmm3
   59775:	vpsllvq %zmm3,%zmm2,%zmm2
   5977b:	vpternlogq $0xc8,%zmm1,%zmm7,%zmm2
   59782:	vpaddq %zmm0,%zmm2,%zmm1
   59788:	kxnorw %k0,%k0,%k1
   5978c:	vpscatterqq %zmm1,(%rdx,%zmm31,8){%k1}
   59793:	vpaddq %zmm15,%zmm30,%zmm1
   59799:	vpsrlq $0x6,%zmm1,%zmm2
   597a0:	vpminuq %zmm4,%zmm2,%zmm3
   597a6:	kxnorw %k0,%k0,%k1
   597aa:	vpxord %xmm31,%xmm31,%xmm31
   597b0:	kxnorw %k0,%k0,%k2
   597b4:	vpgatherqq (%rdi,%zmm2,8),%zmm31{%k1}
   597bb:	vpxor  %xmm2,%xmm2,%xmm2
   597bf:	vpgatherqq (%rax,%zmm3,8),%zmm2{%k2}
   597c6:	vpandq %zmm5,%zmm1,%zmm1
   597cc:	vpsrlvq %zmm1,%zmm31,%zmm1
   597d2:	vpsubq %zmm30,%zmm16,%zmm3
   597d8:	vpaddq %zmm14,%zmm29,%zmm31
   597de:	vpandq %zmm5,%zmm3,%zmm3
   597e4:	vpsllvq %zmm3,%zmm2,%zmm2
   597ea:	vpternlogq $0xc8,%zmm1,%zmm7,%zmm2
   597f1:	vpaddq %zmm0,%zmm2,%zmm1
   597f7:	kxnorw %k0,%k0,%k1
   597fb:	vpscatterqq %zmm1,(%rdx,%zmm31,8){%k1}
   59802:	vpaddq %zmm18,%zmm30,%zmm1
   59808:	vpsrlq $0x6,%zmm1,%zmm2
   5980f:	vpminuq %zmm4,%zmm2,%zmm3
   59815:	kxnorw %k0,%k0,%k1
   59819:	vpxord %xmm31,%xmm31,%xmm31
   5981f:	kxnorw %k0,%k0,%k2
   59823:	vpgatherqq (%rdi,%zmm2,8),%zmm31{%k1}
   5982a:	vpxor  %xmm2,%xmm2,%xmm2
   5982e:	vpgatherqq (%rax,%zmm3,8),%zmm2{%k2}
   59835:	vpandq %zmm5,%zmm1,%zmm1
   5983b:	vpsrlvq %zmm1,%zmm31,%zmm1
   59841:	vpsubq %zmm30,%zmm19,%zmm3
   59847:	vpaddq %zmm17,%zmm29,%zmm31
   5984d:	vpandq %zmm5,%zmm3,%zmm3
   59853:	vpsllvq %zmm3,%zmm2,%zmm2
   59859:	vpternlogq $0xc8,%zmm1,%zmm7,%zmm2
   59860:	vpaddq %zmm0,%zmm2,%zmm1
   59866:	kxnorw %k0,%k0,%k1
   5986a:	vpscatterqq %zmm1,(%rdx,%zmm31,8){%k1}
   59871:	vpaddq %zmm21,%zmm30,%zmm1
   59877:	vpsrlq $0x6,%zmm1,%zmm2
   5987e:	vpminuq %zmm4,%zmm2,%zmm3
   59884:	kxnorw %k0,%k0,%k1
   59888:	vpxord %xmm31,%xmm31,%xmm31
   5988e:	kxnorw %k0,%k0,%k2
   59892:	vpgatherqq (%rdi,%zmm2,8),%zmm31{%k1}
   59899:	vpxor  %xmm2,%xmm2,%xmm2
   5989d:	vpgatherqq (%rax,%zmm3,8),%zmm2{%k2}
   598a4:	vpandq %zmm5,%zmm1,%zmm1
   598aa:	vpsrlvq %zmm1,%zmm31,%zmm1
   598b0:	vpsubq %zmm30,%zmm22,%zmm3
   598b6:	vpaddq %zmm20,%zmm29,%zmm31
   598bc:	vpandq %zmm5,%zmm3,%zmm3
   598c2:	vpsllvq %zmm3,%zmm2,%zmm2
   598c8:	vpternlogq $0xc8,%zmm1,%zmm7,%zmm2
   598cf:	vpaddq %zmm0,%zmm2,%zmm1
   598d5:	kxnorw %k0,%k0,%k1
   598d9:	vpscatterqq %zmm1,(%rdx,%zmm31,8){%k1}
   598e0:	vpaddq %zmm24,%zmm30,%zmm1
   598e6:	vpsrlq $0x6,%zmm1,%zmm2
   598ed:	vpminuq %zmm4,%zmm2,%zmm3
   598f3:	kxnorw %k0,%k0,%k1
   598f7:	vpxord %xmm31,%xmm31,%xmm31
   598fd:	kxnorw %k0,%k0,%k2
   59901:	vpgatherqq (%rdi,%zmm2,8),%zmm31{%k1}
   59908:	vpxor  %xmm2,%xmm2,%xmm2
   5990c:	vpgatherqq (%rax,%zmm3,8),%zmm2{%k2}
   59913:	vpandq %zmm5,%zmm1,%zmm1
   59919:	vpsrlvq %zmm1,%zmm31,%zmm1
   5991f:	vpsubq %zmm30,%zmm25,%zmm3
   59925:	vpaddq %zmm23,%zmm29,%zmm31
   5992b:	vpandq %zmm5,%zmm3,%zmm3
   59931:	vpsllvq %zmm3,%zmm2,%zmm2
   59937:	vpternlogq $0xc8,%zmm1,%zmm7,%zmm2
   5993e:	vpaddq %zmm0,%zmm2,%zmm1
   59944:	kxnorw %k0,%k0,%k1
   59948:	vpscatterqq %zmm1,(%rdx,%zmm31,8){%k1}
   5994f:	vpaddq %zmm27,%zmm30,%zmm1
   59955:	vpsrlq $0x6,%zmm1,%zmm2
   5995c:	vpminuq %zmm4,%zmm2,%zmm3
   59962:	kxnorw %k0,%k0,%k1
   59966:	vpxord %xmm31,%xmm31,%xmm31
   5996c:	kxnorw %k0,%k0,%k2
   59970:	vpgatherqq (%rdi,%zmm2,8),%zmm31{%k1}
   59977:	vpxor  %xmm2,%xmm2,%xmm2
   5997b:	vpgatherqq (%rax,%zmm3,8),%zmm2{%k2}
   59982:	vpandq %zmm5,%zmm1,%zmm1
   59988:	vpsrlvq %zmm1,%zmm31,%zmm1
   5998e:	vpsubq %zmm30,%zmm28,%zmm3
   59994:	vpandq %zmm5,%zmm3,%zmm3
   5999a:	vpsllvq %zmm3,%zmm2,%zmm2
   599a0:	vpternlogq $0xc8,%zmm1,%zmm7,%zmm2
   599a7:	vpaddq %zmm26,%zmm29,%zmm1
   599ad:	vpaddq %zmm0,%zmm2,%zmm2
   599b3:	kxnorw %k0,%k0,%k1
   599b7:	vpscatterqq %zmm2,(%rdx,%zmm1,8){%k1}
   599be:	inc    %rcx
   599c1:	cmp    $0x10,%rcx
   599c5:	jne    59630 <_ZN15funnel_patterns19pat_chunked_aligned17h862079dcf94e24a3E+0x110>
   599cb:	vzeroupper
   599ce:	ret

Disassembly of section .init:

Disassembly of section .fini:

Disassembly of section .plt:
