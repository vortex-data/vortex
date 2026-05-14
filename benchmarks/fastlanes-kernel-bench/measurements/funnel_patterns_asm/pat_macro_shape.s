
/home/user/vortex/target/release/deps/funnel_patterns-b50fc531d4f952e4:     file format elf64-x86-64


Disassembly of section .text:

0000000000059480 <_ZN15funnel_patterns15pat_macro_shape17h4d847f570b7b3821E>:
   59480:	push   %rbx
   59481:	xor    %eax,%eax
   59483:	mov    $0x33,%ecx
   59488:	movabs $0x7fffffffffffe,%r8
   59492:	jmp    594b6 <_ZN15funnel_patterns15pat_macro_shape17h4d847f570b7b3821E+0x36>
   59494:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   594a0:	add    %rsi,%r10
   594a3:	mov    %r10,(%rdx)
   594a6:	add    $0x8,%rdx
   594aa:	add    $0x33,%rax
   594ae:	cmp    $0xcc00,%rax
   594b4:	je     594fa <_ZN15funnel_patterns15pat_macro_shape17h4d847f570b7b3821E+0x7a>
   594b6:	mov    %rax,%r9
   594b9:	shr    $0x6,%r9
   594bd:	mov    %eax,%ebx
   594bf:	and    $0x3f,%ebx
   594c2:	mov    $0x40,%r11d
   594c8:	sub    %ebx,%r11d
   594cb:	cmp    $0x33,%r11d
   594cf:	cmovae %ecx,%r11d
   594d3:	shrx   %rax,(%rdi,%r9,8),%r10
   594d9:	bzhi   %r11,%r10,%r10
   594de:	cmp    $0xe,%ebx
   594e1:	jb     594a0 <_ZN15funnel_patterns15pat_macro_shape17h4d847f570b7b3821E+0x20>
   594e3:	cmp    $0xcbcd,%rax
   594e9:	je     594fc <_ZN15funnel_patterns15pat_macro_shape17h4d847f570b7b3821E+0x7c>
   594eb:	shlx   %r11,0x8(%rdi,%r9,8),%r9
   594f2:	and    %r8,%r9
   594f5:	or     %r9,%r10
   594f8:	jmp    594a0 <_ZN15funnel_patterns15pat_macro_shape17h4d847f570b7b3821E+0x20>
   594fa:	pop    %rbx
   594fb:	ret
   594fc:	lea    0xf95dd(%rip),%rdx        # 152ae0 <anon.41e9924aa43dece028d0278fd7ae1cb2.1.llvm.2134638517557787943+0x18>
   59503:	mov    $0x330,%edi
   59508:	mov    $0x330,%esi
   5950d:	call   *0x1027e5(%rip)        # 15bcf8 <_DYNAMIC+0x338>

Disassembly of section .init:

Disassembly of section .fini:

Disassembly of section .plt:
