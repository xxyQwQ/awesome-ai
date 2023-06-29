	.ORIG	x3000
	LDI	R0,INP		; R0 is target pattern
	LDI	R1,AMT		; R1 is shift counter
LOOP0	BRNZP	LSHIFT		; Do single right shift
CONT0	ADD	R1,R1,#-1	; Decrease shift counter
	BRP	LOOP0		; Repeat certain times
	STI	R0,RES		; Store result to memory
	HALT			; Terminate main part
LSHIFT	AND	R2,R2,#0	; R2 is total pattern
	AND	R3,R3,#0
	ADD	R3,R3,#1	; R3 is mask pattern
	AND	R4,R4,#0	
	ADD	R4,R4,#15	; R4 is bit counter
LOOP1	ADD	R5,R3,R3	; R5 is one-bit higher
	AND	R5,R0,R5
	BRZ	CONT1		; Test higher bit
	ADD	R2,R2,R3	; Accumulate valid bit
CONT1	ADD	R3,R3,R3	; Left shift mask
	ADD	R4,R4,#-1	; Decrease bit counter
	BRP	LOOP1		; Repeat certain times
	ADD	R0,R2,#0	; Store back to R0
	BRNZP	CONT0		; Go back to main part
INP	.FILL	x3100		; x3100 initial pattern
AMT	.FILL	x3101		; x3101 shift amount
RES	.FILL	x3102		; x3102 result storage
	.END