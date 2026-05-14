
/home/user/vortex/target/release/deps/funnel_patterns-b50fc531d4f952e4:     file format elf64-x86-64


Disassembly of section .text:

0000000000059b20 <_ZN15funnel_patterns21pat_combine_then_mask17h1909c53e303e2f8eE>:
   59b20:	push   %rax
   59b21:	add    $0x8,%rdx
   59b25:	mov    $0xffffffffffffffcd,%rax
   59b2c:	mov    $0x33,%ecx
   59b31:	mov    $0x33,%r8b
   59b34:	jmp    59b69 <_ZN15funnel_patterns21pat_combine_then_mask17h1909c53e303e2f8eE+0x49>
   59b36:	cs nopw 0x0(%rax,%rax,1)
   59b40:	shrx   %r10,(%rdi,%r9,8),%r9
   59b46:	bzhi   %r8,%r9,%r9
   59b4b:	add    %rsi,%r9
   59b4e:	mov    %r9,(%rdx)
   59b51:	add    $0xffffffffffffff9a,%rax
   59b55:	add    $0x66,%rcx
   59b59:	add    $0x10,%rdx
   59b5d:	cmp    $0xffffffffffff33cd,%rax
   59b63:	je     59bf0 <_ZN15funnel_patterns21pat_combine_then_mask17h1909c53e303e2f8eE+0xd0>
   59b69:	lea    -0x33(%rcx),%r10
   59b6d:	mov    %r10,%r9
   59b70:	shr    $0x6,%r9
   59b74:	and    $0x3e,%r10d
   59b78:	cmp    $0xe,%r10
   59b7c:	jae    59b90 <_ZN15funnel_patterns21pat_combine_then_mask17h1909c53e303e2f8eE+0x70>
   59b7e:	shrx   %r10,(%rdi,%r9,8),%r9
   59b84:	jmp    59ba8 <_ZN15funnel_patterns21pat_combine_then_mask17h1909c53e303e2f8eE+0x88>
   59b86:	cs nopw 0x0(%rax,%rax,1)
   59b90:	shrx   %r10,(%rdi,%r9,8),%r10
   59b96:	lea    0x33(%rax),%r11d
   59b9a:	and    $0x3e,%r11b
   59b9e:	shlx   %r11,0x8(%rdi,%r9,8),%r9
   59ba5:	or     %r10,%r9
   59ba8:	bzhi   %r8,%r9,%r9
   59bad:	add    %rsi,%r9
   59bb0:	mov    %r9,-0x8(%rdx)
   59bb4:	mov    %rcx,%r9
   59bb7:	shr    $0x6,%r9
   59bbb:	mov    %ecx,%r10d
   59bbe:	and    $0x3f,%r10d
   59bc2:	mov    %ecx,%r11d
   59bc5:	and    $0x3e,%r11d
   59bc9:	cmp    $0xe,%r11d
   59bcd:	jb     59b40 <_ZN15funnel_patterns21pat_combine_then_mask17h1909c53e303e2f8eE+0x20>
   59bd3:	cmp    $0xffffffffffff3433,%rax
   59bd9:	je     59bf2 <_ZN15funnel_patterns21pat_combine_then_mask17h1909c53e303e2f8eE+0xd2>
   59bdb:	shrx   %r10,(%rdi,%r9,8),%r10
   59be1:	shlx   %rax,0x8(%rdi,%r9,8),%r9
   59be8:	or     %r10,%r9
   59beb:	jmp    59b46 <_ZN15funnel_patterns21pat_combine_then_mask17h1909c53e303e2f8eE+0x26>
   59bf0:	pop    %rax
   59bf1:	ret
   59bf2:	lea    0xf8eff(%rip),%rdx        # 152af8 <anon.41e9924aa43dece028d0278fd7ae1cb2.1.llvm.2134638517557787943+0x30>
   59bf9:	mov    $0x330,%edi
   59bfe:	mov    $0x330,%esi
   59c03:	call   *0x1020ef(%rip)        # 15bcf8 <_DYNAMIC+0x338>

Disassembly of section .init:

Disassembly of section .fini:

Disassembly of section .plt:
