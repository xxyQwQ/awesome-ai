; Assembly Calculator
; Computer Architecture (CS2651) Lab 2
; By Xiangyuan Xue (521030910387)
;
;
;
; main procedure
            .ORIG	x3000
            LEA         R0,StartPrompt
            PUTS
            LD          R6,SL1
            ADD         R6,R6,#1        ; R6 is stack pointer
; single command
NewCommand  LEA         R0,InputPrompt
            PUTS                        ; print input prompt
            GETC
            OUT                         ; read and echo input
;
TryExit     LD          R1,EqualX
            ADD         R1,R1,R0
            BRnp        TryDisplay
            JSR         Exit
            HALT
;
TryDisplay  LD          R1,EqualD
            ADD         R1,R1,R0
            BRnp        TryClear
            JSR         Display
            BRnzp       NewCommand
;
TryClear    LD          R1,EqualC
            ADD         R1,R1,R0
            BRnp        TryPlus
            JSR         Clear
            BRnzp       NewCommand
;
TryPlus     LD          R1,EqualPlus
            ADD         R1,R1,R0
            BRnp        TryTime
            JSR         Plus
            BRnzp       NewCommand
;
TryTime     LD          R1,EqualTime
            ADD         R1,R1,R0
            BRnp        TryNega
            JSR         Time
            BRnzp       NewCommand
;
TryNega     LD          R1,EqualNega
            ADD         R1,R1,R0
            BRnp        TryDIV
            JSR         Nega
            BRnzp       NewCommand
;
TryDIV      LD          R1,EqualDIV
            ADD         R1,R1,R0
            BRnp        TryMOD
            JSR         DIV
            BRnzp       NewCommand
;
TryMOD      LD          R1,EqualMOD
            ADD         R1,R1,R0
            BRnp        TryXOR
            JSR         MOD
            BRnzp       NewCommand
;
TryXOR      LD          R1,EqualXOR
            ADD         R1,R1,R0
            BRnp        TryEOL
            JSR         XOR
            BRnzp       NewCommand
;
TryEOL      LD          R1,EqualEOL
            ADD         R1,R1,R0
            BRnp        TryInsert
            BRnzp       NewCommand
;
TryInsert   JSR         Insert
            BRnzp       NewCommand
;
StartPrompt .STRINGZ    "Assembly Calculator by xxyQwQ"
InputPrompt .FILL       x000A
            .STRINGZ    "Command: "
EqualX      .FILL       xFFA8           ; - 'X'
EqualD      .FILL       xFFBC           ; - 'D'
EqualC      .FILL       xFFBD           ; - 'C'
EqualPlus   .FILL       xFFD5           ; - '+'
EqualTime   .FILL       xFFD6           ; - '*'
EqualNega   .FILL       xFFD3           ; - '-'
EqualDIV    .FILL       xFFD1           ; - '/'
EqualMOD    .FILL       xFFDB           ; - '%'
EqualXOR    .FILL       xFFC0           ; - '@'
EqualEOL    .FILL       xFFF6           ; - '\n'
BP1         .FILL       Buffer
SH1         .FILL       StackLimit
SL1         .FILL       StackBase
;
;
;
; exit command
Exit        ST          R0,XSaveR0
            ST          R7,XSaveR7
            LEA         R0,ExitPrompt
            PUTS                        ; print exit prompt
XFinish     LD          R0,XSaveR0
            LD          R7,XSaveR7
            RET
;
XSaveR0     .FILL       x0000
XSaveR7     .FILL       x0000
ExitPrompt  .FILL       x000A
            .STRINGZ    "[Exit]"
;
;
;
; display commnad
Display     ST          R0,DSaveR0
            ST          R5,DSaveR5
            ST          R7,DSaveR7
            JSR         POP             ; fetch value
            ADD         R5,R5,#0
            BRp         DFinish         ; nothing to display
            JSR         B2A
            LD          R0,DNewLine
            OUT                         ; start new line
            LD          R0,BP1
            PUTS                        ; print buffer content
            ADD         R6,R6,#-1       ; push back top value
DFinish     LEA         R0,EchoPrompt
            PUTS                        ; print display prompt
            LD          R0,DSaveR0
            LD          R5,DSaveR5
            LD          R7,DSaveR7
            RET
;
DSaveR0     .FILL       x0000
DSaveR5     .FILL       x0000
DSaveR7     .FILL       x0000
DNewLine    .FILL       x000A
EchoPrompt  .FILL       x000A
            .STRINGZ    "[Display]"
;
;
;
; clear command
Clear       ST          R0,CSaveR0
            ST          R7,CSaveR7
            LD          R6,SL1
            ADD         R6,R6,#1        ; reset stack pointer
            LEA         R0,ClearPrompt
            PUTS                        ; print clear prompt
CFinish     LD          R0,CSaveR0
            LD          R7,CSaveR7
            RET
;
CSaveR0     .FILL       x0000
CSaveR7     .FILL       x0000
ClearPrompt .FILL       x000A
            .STRINGZ    "[Clear]"
;
;
;
; check whether value in R0 between -999 and +999, and R5 report check state
RangeCheck  ST          R0,RCSaveR0
            ST          R7,RCSaveR7
            LD          R5,Minus999
            ADD         R5,R5,R0
            BRp         RCFail
            LD          R5,Plus999
            ADD         R5,R5,R0
            BRn         RCFail
            AND         R5,R5,#0
            BRnzp       RCFinish
RCFail      LEA         R0,RangeError
            PUTS
            AND         R5,R5,#0
            ADD         R5,R5,#1
RCFinish    LD          R0,RCSaveR0
            LD          R7,RCSaveR7
            RET
RCSaveR0    .FILL       x0000
RCSaveR7    .FILL       x0000
Plus999     .FILL       #999
Minus999    .FILL       #-999
BP2         .FILL       Buffer
SH2         .FILL       StackLimit
SL2         .FILL       StackBase
;
;
;
; plus command
Plus        ST          R0,PlusSaveR0
            ST          R1,PlusSaveR1
            ST          R5,PlusSaveR5
            ST          R7,PlusSaveR7
            JSR         POP             ; pop first element
            ADD         R5,R5,#0
            BRp         PlusAbsent      ; fail to pop
            ADD         R1,R0,#0
            JSR         POP             ; pop second element
            ADD         R5,R5,#0
            BRp         PlusBack        ; fail to pop
            ADD         R0,R0,R1
            JSR         RangeCheck      ; check result
            ADD         R5,R5,#0
            BRp         PlusExceed      ; range exceeds
            JSR         PUSH            ; push result to stack
            BRnzp       PlusFinish
PlusExceed  ADD         R6,R6,#-1
            ADD         R6,R6,#-1
            BRnzp       PlusFinish
PlusBack    ADD         R6,R6,#-1
PlusAbsent  LEA         R0,CountError
            PUTS                        ; print error prompt
PlusFinish  LEA         R0,PlusPrompt
            PUTS                        ; print plus prompt
            LD          R0,PlusSaveR0
            LD          R1,PlusSaveR1
            LD          R5,PlusSaveR5
            LD          R7,PlusSaveR7
            RET
;
PlusSaveR0  .FILL       x0000
PlusSaveR1  .FILL       x0000
PlusSaveR5  .FILL       x0000
PlusSaveR7  .FILL       x0000
PlusPrompt  .FILL       x000A
            .STRINGZ    "[Plus]"
;
;
;
; multiply command
Time        ST          R0,TimeSaveR0
            ST          R1,TimeSaveR1
            ST          R2,TimeSaveR2
            ST          R3,TimeSaveR3
            ST          R5,TimeSaveR5
            ST          R7,TimeSaveR7
            JSR         POP             ; pop first element
            ADD         R5,R5,#0
            BRp         TimeAbsent      ; fail to pop
            ADD         R1,R0,#0
            JSR         POP             ; pop second element
            ADD         R5,R5,#0
            BRp         TimeBack        ; fail to pop
            AND         R3,R3,#0        ; R3 is flip label
            ADD         R2,R0,#0
            BRzp        TimeStart       ; operand is not negative
            ADD         R3,R3,#1
            NOT         R2,R2
            ADD         R2,R2,#1        ; operand is negative
TimeStart   AND         R0,R0,#0
            ADD         R2,R2,#0
            BRz         TimePush        ; multiply zero
TimeLoop    ADD         R0,R0,R1
            ADD         R2,R2,#-1
            BRp         TimeLoop        ; accumulate by loop
            JSR         RangeCheck      ; check result
            ADD         R5,R5,#0
            BRp         TimeExceed      ; range exceeds
            ADD         R3,R3,#0
            BRz         TimePush        ; keep result
            NOT         R0,R0
            ADD         R0,R0,#1        ; negate result
TimePush    JSR         PUSH            ; push result to stack
            BRnzp       TimeFinish
TimeExceed  ADD         R6,R6,#-1
            ADD         R6,R6,#-1
            BRnzp       TimeFinish
TimeBack    ADD         R6,R6,#-1
TimeAbsent  LEA         R0,CountError
            PUTS                        ; print error prompt
TimeFinish  LEA         R0,TimePrompt
            PUTS                        ; print multiply prompt
            LD          R0,TimeSaveR0
            LD          R1,TimeSaveR1
            LD          R2,TimeSaveR2
            LD          R3,TimeSaveR3
            LD          R5,TimeSaveR5
            LD          R7,TimeSaveR7
            RET
TimeSaveR0  .FILL       x0000
TimeSaveR1  .FILL       x0000
TimeSaveR2  .FILL       x0000
TimeSaveR3  .FILL       x0000
TimeSaveR5  .FILL       x0000
TimeSaveR7  .FILL       x0000
TimePrompt  .FILL       x000A
            .STRINGZ    "[Multiply]"
;
;
;
; negate command
Nega        ST          R0,NegaSaveR0
            ST          R5,NegaSaveR5
            ST          R7,NegaSaveR7
            JSR         POP             ; pop top element
            ADD         R5,R5,#0
            BRp         NegaAbsent      ; fail to pop
            NOT         R0,R0
            ADD         R0,R0,#1        ; negate result
            JSR         PUSH            ; push result to stack
            BRnzp       NegaFinish
NegaAbsent  LEA         R0,CountError
            PUTS                        ; print error prompt
NegaFinish  LEA         R0,NegaPrompt
            PUTS                        ; print negate prompt
            LD          R0,NegaSaveR0
            LD          R5,NegaSaveR5
            LD          R7,NegaSaveR7
            RET
NegaSaveR0  .FILL       x0000
NegaSaveR5  .FILL       x0000
NegaSaveR7  .FILL       x0000
NegaPrompt  .FILL       x000A
            .STRINGZ    "[Negate]"
;
;
;
; divide command
DIV         ST          R0,DIVSaveR0
            ST          R1,DIVSaveR1
            ST          R2,DIVSaveR2
            ST          R5,DIVSaveR5
            ST          R7,DIVSaveR7
            JSR         POP             ; pop first element
            ADD         R5,R5,#0
            BRp         DIVAbsent       ; fail to pop
            ADD         R1,R0,#0
            JSR         POP             ; pop second element
            ADD         R5,R5,#0
            BRp         DIVBack         ; fail to pop
            ADD         R2,R0,#0
            BRz         DIVIllegal      ; divided by zero
            BRp         DIVStart        ; divisor is positive
            NOT         R1,R1
            ADD         R1,R1,#1
            NOT         R2,R2
            ADD         R2,R2,#1        ; flip dividend and divisor
DIVStart    AND         R0,R0,#0
            ADD         R1,R1,#0
            BRp         DIVPlus         ; dividend is positive
            BRn         DIVMinus        ; dividend is negative
            BRnzp       DIVPush         ; dividend is zero
DIVPlus     NOT         R2,R2
            ADD         R2,R2,#1        ; flip divisor
DIVLoop     ADD         R1,R1,R2
            BRn         DIVPush
            ADD         R0,R0,#1
            BRnzp       DIVLoop         ; accumulate by loop
DIVMinus    ADD         R0,R0,#-1
            ADD         R1,R1,R2
            BRzp        DIVPush
            BRnzp       DIVMinus        ; accumulate by loop
DIVPush     JSR         PUSH
            BRnzp       DIVFinish       ; push result to stack
DIVIllegal  LEA         R0,DivideZero
            PUTS                        ; print error prompt
            ADD         R6,R6,#-1
            ADD         R6,R6,#-1
            BRnzp       DIVFinish
DIVBack     ADD         R6,R6,#-1
DIVAbsent   LEA         R0,CountError
            PUTS                        ; print error prompt
DIVFinish   LEA         R0,DIVPrompt
            PUTS                        ; print divide prompt
            LD          R0,DIVSaveR0
            LD          R1,DIVSaveR1
            LD          R2,DIVSaveR2
            LD          R5,DIVSaveR5
            LD          R7,DIVSaveR7
            RET
;
CountError  .FILL       x000A
            .STRINGZ    "[Error] not enough operands"
RangeError  .FILL       x000A
            .STRINGZ    "[Error] range limit exceeds"
DIVSaveR0   .FILL       x0000
DIVSaveR1   .FILL       x0000
DIVSaveR2   .FILL       x0000
DIVSaveR5   .FILL       x0000
DIVSaveR7   .FILL       x0000
DIVPrompt   .FILL       x000A
            .STRINGZ    "[Divide]"
DivideZero  .FILL       x000A
            .STRINGZ    "[Error] divide by zero"
;
;
;
; modular command
MOD         ST          R0,MODSaveR0
            ST          R1,MODSaveR1
            ST          R2,MODSaveR2
            ST          R5,MODSaveR5
            ST          R7,MODSaveR7
            JSR         POP             ; pop first element
            ADD         R5,R5,#0
            BRp         MODAbsent       ; fail to pop
            ADD         R1,R0,#0
            JSR         POP             ; pop second element
            ADD         R5,R5,#0
            BRp         MODBack         ; fail to pop
            ADD         R2,R0,#0
            BRz         MODIllegal      ; divided by zero
            BRp         MODStart        ; divisor is positive
            NOT         R1,R1
            ADD         R1,R1,#1
            NOT         R2,R2
            ADD         R2,R2,#1        ; flip dividend and divisor
MODStart    AND         R0,R0,#0
            ADD         R1,R1,#0
            BRp         MODPlus         ; dividend is positive
            BRn         MODMinus        ; dividend is negative
            BRnzp       MODPush         ; dividend is zero
MODPlus     NOT         R2,R2
            ADD         R2,R2,#1        ; flip divisor
MODLoop     ADD         R0,R1,R2
            BRn         MODPush
            ADD         R1,R0,#0
            BRnzp       MODLoop         ; test by loop
MODMinus    ADD         R1,R1,R2
            BRzp        MODPush
            BRnzp       MODMinus        ; test by loop
MODPush     ADD         R0,R1,#0
            JSR         PUSH
            BRnzp       MODFinish       ; push result to stack
MODIllegal  LEA         R0,DivideZero
            PUTS                        ; print error prompt
            ADD         R6,R6,#-1
            ADD         R6,R6,#-1
            BRnzp       MODFinish
MODBack     ADD         R6,R6,#-1
MODAbsent   LEA         R0,CountError
            PUTS                        ; print error prompt
MODFinish   LEA         R0,MODPrompt
            PUTS                        ; print modular prompt
            LD          R0,MODSaveR0
            LD          R1,MODSaveR1
            LD          R2,MODSaveR2
            LD          R5,MODSaveR5
            LD          R7,MODSaveR7
            RET
MODSaveR0   .FILL       x0000
MODSaveR1   .FILL       x0000
MODSaveR2   .FILL       x0000
MODSaveR5   .FILL       x0000
MODSaveR7   .FILL       x0000
MODPrompt   .FILL       x000A
            .STRINGZ    "[Modular]"
;
;
;
; exlude command
XOR         ST          R0,XORSaveR0
            ST          R1,XORSaveR1
            ST          R2,XORSaveR2
            ST          R5,XORSaveR5
            ST          R7,XORSaveR7
            JSR         POP             ; pop first element
            ADD         R5,R5,#0
            BRp         XORAbsent       ; fail to pop
            ADD         R1,R0,#0
            JSR         POP             ; pop second element
            ADD         R5,R5,#0
            BRp         XORBack         ; fail to pop
            NOT         R2,R0
            AND         R2,R2,R1
            NOT         R2,R2
            NOT         R5,R1
            AND         R5,R5,R0
            NOT         R5,R5
            AND         R0,R2,R5
            NOT         R0,R0           ; exclude operation
XORPush     JSR         PUSH
            BRnzp       XORFinish
XORBack     ADD         R6,R6,#-1
XORAbsent   LEA         R0,CountError
            PUTS                        ; print error prompt
XORFinish   LEA         R0,XORPrompt
            PUTS                        ; print exlude prompt
            LD          R0,XORSaveR0
            LD          R1,XORSaveR1
            LD          R2,XORSaveR2
            LD          R5,XORSaveR5
            LD          R7,XORSaveR7
            RET
;
XORSaveR0   .FILL       x0000
XORSaveR1   .FILL       x0000
XORSaveR2   .FILL       x0000
XORSaveR5   .FILL       x0000
XORSaveR7   .FILL       x0000
XORPrompt   .FILL       x000A
            .STRINGZ    "[Exclude]"
;
;
;
; insert command
Insert      ST          R0,ISaveR0
            ST          R1,ISaveR1
            ST          R2,ISaveR2
            ST          R3,ISaveR3
            ST          R5,ISaveR5
            ST          R7,ISaveR7
            LD          R1,BP3
            LD          R2,DigitLimit
ILoop       LD          R3,EqualEnter
            ADD         R3,R3,R0
            BRz         ILegal          ; finish input
            ADD         R2,R2,#0
            BRz         IIllegal        ; too large input
            LD          R3,EqualZero
            ADD         R3,R3,R0
            BRn         ICrazy
            LD          R3,EqualNine
            ADD         R3,R3,R0
            BRp         ICrazy          ; check datatype
            ADD         R2,R2,#-1
            STR         R0,R1,#0        ; store digit into buffer
            ADD         R1,R1,#1        ; adjust buffer pointer
            GETC
            OUT                         ; input next digit
            BRnzp       ILoop
ICrazy      GETC
            OUT
            LD          R3,EqualEnter
            ADD         R3,R3,R0
            BRnp        ICrazy          ; read till the end
            LEA         R0,CrazyPrompt
            PUTS
            LD          R0,INewLine
            OUT
            BRnzp       IFinish
ILegal      LD          R2,BP3
            NOT         R2,R2
            ADD         R2,R2,#1
            ADD         R1,R1,R2        ; count size in buffer
            JSR         A2B
            JSR         PUSH            ; push value into stack
            BRnzp       IFinish
IIllegal    GETC
            OUT
            LD          R3,EqualEnter
            ADD         R3,R3,R0
            BRnp        IIllegal        ; read till the end
            LEA         R0,LargePrompt
            PUTS                        ; print error prompt
            LD          R0,INewLine
            OUT
IFinish     LEA         R0,ValuePrompt
            PUTS                        ; print insert prompt
            LD          R0,ISaveR0
            LD          R1,ISaveR1
            LD          R2,ISaveR2
            LD          R3,ISaveR3
            LD          R5,ISaveR5
            LD          R7,ISaveR7
            RET
;
ISaveR0     .FILL       x0000
ISaveR1     .FILL       x0000
ISaveR2     .FILL       x0000
ISaveR3     .FILL       x0000
ISaveR5     .FILL       x0000
ISaveR7     .FILL       x0000
DigitLimit  .FILL       #3
EqualEnter  .FILL       xFFF6
EqualZero   .FILL       x-30
EqualNine   .FILL       x-39
INewLine    .FILL       x000A
CrazyPrompt .STRINGZ    "[Error] not an integer"
LargePrompt .STRINGZ    "[Error] too large value"
ValuePrompt .STRINGZ    "[Insert]"
BP3         .FILL       Buffer
SH3         .FILL       StackLimit
SL3         .FILL       StackBase
;
;
;
; pop top element to R0, and R5 report pop state
POP         ST          R7,POPSaveR7
            LD          R0,SL3
            ADD         R0,R0,#1
            NOT         R0,R0
            ADD         R0,R0,#1        ; empty pointer
            ADD         R0,R0,R6
            BRz         UnderFlow
POPValid    LDR         R0,R6,#0        ; fetch value
            ADD         R6,R6,#1        ; adjust stack pointer
            AND         R5,R5,#0        ; report success
            BRnzp       POPFinish
UnderFlow   LEA         R0,UnderPrompt
            PUTS                        ; print error prompt
            AND         R5,R5,#0
            ADD         R5,R5,#1        ; report failure
POPFinish   LD          R7,POPSaveR7
            RET
;
POPSaveR7   .FILL       x0000
UnderPrompt .FILL       x000A
            .STRINGZ    "[Error] stack is empty"
;
;
;
; push R0 to stack, and R5 report push state
PUSH        ST          R0,PUSHSaveR0
            ST          R1,PUSHSaveR1
            ST          R7,PUSHSaveR7
            LD          R1,SH3
            NOT         R1,R1
            ADD         R1,R1,#1        ; full pointer
            ADD         R1,R1,R6
            BRz         OverFlow
PUSHValid   ADD         R6,R6,#-1       ; adjust stack pointer
            STR         R0,R6,#0        ; store value
            AND         R5,R5,#0        ; report success
            BRnzp       PUSHFinish
OverFlow    LEA         R0,OverPrompt
            PUTS                        ; print error prompt
            LD          R0,PUSHNewLine
            OUT
            AND         R5,R5,#0
            ADD         R5,R5,#1        ; report failure
PUSHFinish  LD          R0,PUSHSaveR0
            LD          R1,PUSHSaveR1
            LD          R7,PUSHSaveR7
            RET
;
PUSHSaveR0  .FILL       x0000
PUSHSaveR1  .FILL       x0000
PUSHSaveR7  .FILL       x0000
PUSHNewLine .FILL       x000A
OverPrompt  .STRINGZ    "[Error] stack is full"
;
;
;
; convert ascii in buffer to binary in R0, and R1 counts number of digits
A2B         ST          R1,A2BSaveR1
            ST          R2,A2BSaveR2
            ST          R3,A2BSaveR3
            ST          R4,A2BSaveR4
            ST          R5,A2BSaveR5
            ST          R7,A2BSaveR7
            AND         R0,R0,#0
            ADD         R1,R1,#0
            BRz         A2BFinish       ; 0 digit
            LD          R3,A2BOffset
            LD          R2,BP3
A2BBit1     ADD         R1,R1,#-1
            ADD         R2,R2,R1
            LDR         R4,R2,#0        ; load digit
            ADD         R4,R4,R3        ; convert character to integer
            ADD         R0,R0,R4        ; digit weight is 1
            ADD         R1,R1,#0
            BRz         A2BFinish       ; 1 digit
A2BBit10    ADD         R1,R1,#-1
            ADD         R2,R2,#-1
            LDR         R4,R2,#0        ; load digit
            ADD         R4,R4,R3        ; convert character to integer
            LEA         R5,Time10
            ADD         R5,R5,R4
            LDR         R4,R5,#0
            ADD         R0,R0,R4        ; digit weight is 10
            ADD         R1,R1,#0
            BRz         A2BFinish       ; 2 digits
A2BBit100   ADD         R2,R2,#-1
            LDR         R4,R2,#0        ; load digit
            ADD         R4,R4,R3        ; convert character to integer
            LEA         R5,Time100
            ADD         R5,R5,R4
            LDR         R4,R5,#0
            ADD         R0,R0,R4        ; digit weight is 100
A2BFinish   LD          R1,A2BSaveR1
            LD          R2,A2BSaveR2
            LD          R3,A2BSaveR3
            LD          R4,A2BSaveR4
            LD          R5,A2BSaveR5
            LD          R7,A2BSaveR7
            RET
;
A2BSaveR1   .FILL       x0000
A2BSaveR2   .FILL       x0000
A2BSaveR3   .FILL       x0000
A2BSaveR4   .FILL       x0000
A2BSaveR5   .FILL       x0000
A2BSaveR7   .FILL       x0000
A2BOffset   .FILL       xFFD0
;
;
;
; convert binary in R0 to ascii in buffer
B2A         ST          R0,B2ASaveR0
            ST          R1,B2ASaveR1
            ST          R2,B2ASaveR2
            ST          R3,B2ASaveR3
            ST          R7,B2ASaveR7
            LD          R1,BP3
            ADD         R0,R0,#0
            BRn         B2ASign
            LD          R2,ASCIIPlus
            STR         R2,R1,#0
            BRnzp       B2ABegin100
B2ASign     LD          R2,ASCIIMinus
            STR         R2,R1,#0
            NOT         R0,R0
            ADD         R0,R0,#1
B2ABegin100 LD          R2,B2AOffset
            LD          R3,Minus100
B2ALoop100  ADD         R0,R0,R3
            BRn         B2AEnd100       ; count finish
            ADD         R2,R2,#1
            BRnzp       B2ALoop100
B2AEnd100   STR         R2,R1,#1        ; digit weight is 100
            LD          R3,Plus100
            ADD         R0,R0,R3        ; add 100 back
B2ABegin10  LD          R2,B2AOffset
B2ALoop10   ADD         R0,R0,#-10
            BRn         B2AEnd10        ; count finish
            ADD         R2,R2,#1
            BRnzp       B2ALoop10
B2AEnd10    STR         R2,R1,#2        ; digit weight is 10
            ADD         R0,R0,#10       ; add 10 back
B2ALast1    LD          R2,B2AOffset
            ADD         R2,R2,R0
            STR         R2,R1,#3        ; digit weight is 1
B2AFinish   LD          R0,B2ASaveR0
            LD          R1,B2ASaveR1
            LD          R2,B2ASaveR2
            LD          R3,B2ASaveR3
            LD          R7,B2ASaveR7
            RET
;
B2ASaveR0   .FILL       x0000
B2ASaveR1   .FILL       x0000
B2ASaveR2   .FILL       x0000
B2ASaveR3   .FILL       x0000
B2ASaveR7   .FILL       x0000
B2AOffset   .FILL       x0030
ASCIIPlus   .FILL       x002B
ASCIIMinus  .FILL       x002D
Plus100     .FILL       #100
Minus100    .FILL       #-100
Time10      .FILL       #0
            .FILL       #10
            .FILL       #20
            .FILL       #30
            .FILL       #40
            .FILL       #50
            .FILL       #60
            .FILL       #70
            .FILL       #80
            .FILL       #90
Time100     .FILL       #0
            .FILL       #100
            .FILL       #200
            .FILL       #300
            .FILL       #400
            .FILL       #500
            .FILL       #600
            .FILL       #700
            .FILL       #800
            .FILL       #900
;
;
;
; program stack and buffer
StackLimit  .BLKW       #15
StackBase   .FILL       x0000           ; reserve stack space
Buffer      .BLKW       #4
            .FILL       X0000
            .END