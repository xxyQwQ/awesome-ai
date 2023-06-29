#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
typedef int16_t DATA;
typedef uint16_t uDATA;

// To be realized
void process_instruction();

// A couple of useful definitions.
#define FALSE 0
#define TRUE 1

// Use this to avoid overflowing 16 bits on the bus.
#define To32bits(x) ((x) | (0x00000000))
#define Low16bits(x) ((x) & (0xFFFF))
#define Low3bits(x) ((x) & (0b111))

// Main memory.
#define WORDS_IN_MEM 0x08000
int MEMORY[WORDS_IN_MEM]; // MEMORY[A] stores the word address A

// LC-3 State info.
#define LC_3_REGS 8
int RUN_BIT; // run bit
typedef struct System_Latches_Struct
{
	int PC,				 // program counter
		N,				 // n condition bit
		Z,				 // z condition bit
		P;				 // p condition bit
	int REGS[LC_3_REGS]; // register file
} System_Latches;

// Data Structure for Latch.
System_Latches CURRENT_LATCHES, NEXT_LATCHES;

// A cycle counter.
int INSTRUCTION_COUNT;

// Procedure: help
// Purpose: Print out a list of commands
void help()
{
	printf("----------------LC-3 ISIM Help-----------------------\n");
	printf("go               -  run program to completion         \n");
	printf("run n            -  execute program for n instructions\n");
	printf("mdump low high   -  dump memory from low to high      \n");
	printf("rdump            -  dump the register & bus values    \n");
	printf("?                -  display this help menu            \n");
	printf("quit             -  exit the program                  \n\n");
}

// Procedure: cycle
// Purpose: Execute a cycle
void cycle()
{
	process_instruction();
	CURRENT_LATCHES = NEXT_LATCHES;
	INSTRUCTION_COUNT++;
}

// Procedure: run
// Purpose: Simulate the LC-3 for n cycles
void run(int num_cycles)
{
	int i;
	if (RUN_BIT == FALSE)
	{
		printf("Can't simulate, Simulator is halted\n\n");
		return;
	}
	printf("Simulating for %d cycles...\n\n", num_cycles);
	for (i = 0; i < num_cycles; i++)
	{
		if (CURRENT_LATCHES.PC == 0x0000)
		{
			RUN_BIT = FALSE;
			printf("Simulator halted\n\n");
			break;
		}
		cycle();
	}
}

// Procedure: go
// Purpose: Simulate the LC-3 until HALTed
void go()
{
	if (RUN_BIT == FALSE)
	{
		printf("Can't simulate, Simulator is halted\n\n");
		return;
	}
	printf("Simulating...\n\n");
	while (CURRENT_LATCHES.PC != 0x0000)
		cycle();
	RUN_BIT = FALSE;
	printf("Simulator halted\n\n");
}

// Procedure: mdump
// Purpose: Dump a word-aligned region of memory to the output file
void mdump(FILE *dumpsim_file, int start, int stop)
{
	int address;
	printf("\nMemory content [0x%.4x..0x%.4x] :\n", start, stop);
	printf("-------------------------------------\n");
	for (address = start; address <= stop; address++)
		printf("  0x%.4x (%d) : 0x%.2x\n", address, address, MEMORY[address]);
	printf("\n");
	fprintf(dumpsim_file, "\nMemory content [0x%.4x..0x%.4x] :\n", start, stop);
	fprintf(dumpsim_file, "-------------------------------------\n");
	for (address = start; address <= stop; address++)
		fprintf(dumpsim_file, " 0x%.4x (%d) : 0x%.2x\n", address, address, MEMORY[address]);
	fprintf(dumpsim_file, "\n");
	fflush(dumpsim_file);
}

// Procedure: rdump
// Purpose: Dump current register and bus values to the output file
void rdump(FILE *dumpsim_file)
{
	int k;
	printf("\nCurrent register/bus values :\n");
	printf("-------------------------------------\n");
	printf("Instruction Count : %d\n", INSTRUCTION_COUNT);
	printf("PC                : 0x%.4x\n", CURRENT_LATCHES.PC);
	printf("CCs: N = %d  Z = %d  P = %d\n", CURRENT_LATCHES.N, CURRENT_LATCHES.Z, CURRENT_LATCHES.P);
	printf("Registers:\n");
	for (k = 0; k < LC_3_REGS; k++)
		printf("%d: 0x%.4x\n", k, CURRENT_LATCHES.REGS[k]);
	printf("\n");
	fprintf(dumpsim_file, "\nCurrent register/bus values :\n");
	fprintf(dumpsim_file, "-------------------------------------\n");
	fprintf(dumpsim_file, "Instruction Count : %d\n", INSTRUCTION_COUNT);
	fprintf(dumpsim_file, "PC                : 0x%.4x\n", CURRENT_LATCHES.PC);
	fprintf(dumpsim_file, "CCs: N = %d  Z = %d  P = %d\n", CURRENT_LATCHES.N, CURRENT_LATCHES.Z, CURRENT_LATCHES.P);
	fprintf(dumpsim_file, "Registers:\n");
	for (k = 0; k < LC_3_REGS; k++)
		fprintf(dumpsim_file, "%d: 0x%.4x\n", k, CURRENT_LATCHES.REGS[k]);
	fprintf(dumpsim_file, "\n");
	fflush(dumpsim_file);
}

// Procedure: get_command
// Purpose: Read a command from standard input
void get_command(FILE *dumpsim_file)
{
	char buffer[20];
	int start, stop, cycles;
	printf("LC-3-SIM> ");
	scanf("%s", buffer);
	printf("\n");
	switch (buffer[0])
	{
	case 'G':
	case 'g':
		go();
		break;
	case 'M':
	case 'm':
		scanf("%i %i", &start, &stop);
		mdump(dumpsim_file, start, stop);
		break;
	case '?':
		help();
		break;
	case 'Q':
	case 'q':
		printf("Bye.\n");
		exit(0);
	case 'R':
	case 'r':
		if (buffer[1] == 'd' || buffer[1] == 'D')
			rdump(dumpsim_file);
		else
			scanf("%d", &cycles), run(cycles);
		break;
	default:
		printf("Invalid Command\n");
		break;
	}
}

// Procedure: init_memory
// Purpose: Zero out the memory array
void init_memory()
{
	int i;
	for (i = 0; i < WORDS_IN_MEM; i++)
		MEMORY[i] = 0;
}

// Procedure: load_program
// Purpose: Load program and service routines into memory
void load_program(char *program_filename)
{
	FILE *prog;
	int ii, word, program_base;
	prog = fopen(program_filename, "r");
	if (prog == NULL)
	{
		printf("Error: Can't open program file %s\n", program_filename);
		exit(-1);
	}
	if (fscanf(prog, "%x\n", &word) != EOF)
		program_base = word;
	else
	{
		printf("Error: Program file is empty\n");
		exit(-1);
	}
	ii = 0;
	while (fscanf(prog, "%x\n", &word) != EOF)
	{
		if (program_base + ii >= WORDS_IN_MEM)
		{
			printf("Error: Program file %s is too long to fit in memory. %x\n", program_filename, ii);
			exit(-1);
		}
		MEMORY[program_base + ii] = word;
		ii++;
	}
	if (CURRENT_LATCHES.PC == 0)
		CURRENT_LATCHES.PC = program_base;
	printf("Read %d words from program into memory.\n\n", ii);
}

// Procedure: initialize
// Purpose: Load machine language program and set up initial state of the machine
void initialize(char *program_filename, int num_prog_files)
{
	int i;
	init_memory();
	for (i = 0; i < num_prog_files; i++)
	{
		load_program(program_filename);
		while (*program_filename++ != '\0')
			;
	}
	CURRENT_LATCHES.Z = 1;
	NEXT_LATCHES = CURRENT_LATCHES;
	RUN_BIT = TRUE;
}

// Procedure: main
int main(int argc, char *argv[])
{
	FILE *dumpsim_file;
	if (argc < 2)
	{
		printf("Error: usage: %s <program_file_1> <program_file_2> ...\n", argv[0]);
		exit(1);
	}
	printf("LC-3 Simulator\n\n");
	initialize(argv[1], argc - 1);
	if ((dumpsim_file = fopen("dumpsim", "w")) == NULL)
	{
		printf("Error: Can't open dumpsim file\n");
		exit(-1);
	}
	while (1)
		get_command(dumpsim_file);
	return 0;
}

// Self-written codes
typedef enum
{
	OPCODE_ADD = 0b0001,
	OPCODE_AND = 0b0101,
	OPCODE_BR = 0b0000,
	OPCODE_JMP = 0b1100,
	OPCODE_JSR = 0b0100,
	OPCODE_LD = 0b0010,
	OPCODE_LDI = 0b1010,
	OPCODE_LDR = 0b0110,
	OPCODE_LEA = 0b1110,
	OPCODE_NOT = 0b1001,
	OPCODE_RTI = 0b1000,
	OPCODE_ST = 0b0011,
	OPCODE_STI = 0b1011,
	OPCODE_STR = 0b0111,
	OPCODE_TRAP = 0b1111,
	OPCODE_RESERVED = 0b1101,
} OPCODE;

DATA sext(DATA value, DATA bits)
{
	if (value & (1 << (bits - 1)))
	{
		DATA mask = 0xFFFF ^ ((1 << bits) - 1);
		return value | mask;
	}
	return value;
}

void setcc(DATA value)
{
	if (value == 0)
	{
		NEXT_LATCHES.N = 0;
		NEXT_LATCHES.Z = 1;
		NEXT_LATCHES.P = 0;
	}
	else if (value > 0)
	{
		NEXT_LATCHES.N = 0;
		NEXT_LATCHES.Z = 0;
		NEXT_LATCHES.P = 1;
	}
	else
	{
		NEXT_LATCHES.N = 1;
		NEXT_LATCHES.Z = 0;
		NEXT_LATCHES.P = 0;
	}
}

void process_instruction()
{
	NEXT_LATCHES = CURRENT_LATCHES;
	DATA PROGRAM_COUNTER = Low16bits(NEXT_LATCHES.PC);
	uDATA IR = Low16bits(MEMORY[PROGRAM_COUNTER]);
	PROGRAM_COUNTER = PROGRAM_COUNTER + 1;
	NEXT_LATCHES.PC = PROGRAM_COUNTER;
	OPCODE OP = (IR >> 12);

	switch (OP)
	{
	case OPCODE_ADD:
	{
		uDATA DR = Low3bits(IR >> 9);
		DATA RESULT;
		uDATA SR1 = Low3bits(IR >> 6);
		DATA OPERAND1 = Low16bits(NEXT_LATCHES.REGS[SR1]);
		if (IR & (1 << 5)) // immediate mode
		{
			DATA IMM5 = sext((IR & ((1 << 5) - 1)), 5);
			RESULT = OPERAND1 + IMM5;
		}
		else // register mode
		{
			uDATA SR2 = Low3bits(IR);
			DATA OPERAND2 = Low16bits(NEXT_LATCHES.REGS[SR2]);
			RESULT = OPERAND1 + OPERAND2;
		}
		NEXT_LATCHES.REGS[DR] = To32bits(RESULT);
		setcc(RESULT);
		break;
	}

	case OPCODE_AND:
	{
		uDATA DR = Low3bits(IR >> 9);
		DATA RESULT;
		uDATA SR1 = Low3bits(IR >> 6);
		DATA OPERAND1 = Low16bits(NEXT_LATCHES.REGS[SR1]);
		if (IR & (1 << 5)) // immediate mode
		{
			DATA IMM5 = sext((IR & ((1 << 5) - 1)), 5);
			RESULT = OPERAND1 & IMM5;
		}
		else // register mode
		{
			uDATA SR2 = Low3bits(IR);
			DATA OPERAND2 = Low16bits(NEXT_LATCHES.REGS[SR2]);
			RESULT = OPERAND1 & OPERAND2;
		}
		NEXT_LATCHES.REGS[DR] = To32bits(RESULT);
		setcc(RESULT);
		break;
	}

	case OPCODE_BR:
	{
		uDATA NZP = Low3bits(IR >> 9);
		uDATA STATE = Low16bits((NEXT_LATCHES.N << 2) | (NEXT_LATCHES.Z << 1) | (NEXT_LATCHES.P));
		if (NZP & STATE) // condition satisfied
		{
			DATA OFFSET9 = sext((IR & ((1 << 9) - 1)), 9);
			DATA PC = Low16bits(NEXT_LATCHES.PC);
			PC += OFFSET9;
			NEXT_LATCHES.PC = To32bits(PC);
		}
		break;
	}

	case OPCODE_JMP:
	{
		uDATA BASER = Low3bits(IR >> 6);
		DATA PC = Low16bits(NEXT_LATCHES.REGS[BASER]);
		NEXT_LATCHES.PC = To32bits(PC);
		break;
	}

	case OPCODE_JSR:
	{
		DATA PC = Low16bits(NEXT_LATCHES.PC);
		NEXT_LATCHES.REGS[7] = PC;
		if (IR & (1 << 11)) // type JSR
		{
			DATA OFFSET11 = sext((IR & ((1 << 11) - 1)), 11);
			PC += OFFSET11;
			NEXT_LATCHES.PC = To32bits(PC);
		}
		else // type JSRR
		{
			uDATA BASER = Low3bits(IR >> 6);
			PC = Low16bits(NEXT_LATCHES.REGS[BASER]);
			NEXT_LATCHES.PC = To32bits(PC);
		}
		break;
	}

	case OPCODE_LD:
	{
		uDATA DR = Low3bits(IR >> 9);
		DATA OFFSET9 = sext((IR & ((1 << 9) - 1)), 9);
		DATA PC = Low16bits(NEXT_LATCHES.PC);
		DATA ADDRESS = PC + OFFSET9;
		DATA RESULT = Low16bits(MEMORY[ADDRESS]);
		NEXT_LATCHES.REGS[DR] = To32bits(RESULT);
		setcc(RESULT);
		break;
	}

	case OPCODE_LDI:
	{
		uDATA DR = Low3bits(IR >> 9);
		DATA OFFSET9 = sext((IR & ((1 << 9) - 1)), 9);
		DATA PC = Low16bits(NEXT_LATCHES.PC);
		DATA ADDRESS = PC + OFFSET9;
		ADDRESS = Low16bits(MEMORY[ADDRESS]);
		DATA RESULT = Low16bits(MEMORY[ADDRESS]);
		NEXT_LATCHES.REGS[DR] = To32bits(RESULT);
		setcc(RESULT);
		break;
	}

	case OPCODE_LDR:
	{
		uDATA DR = Low3bits(IR >> 9);
		uDATA BASER = Low3bits(IR >> 6);
		DATA BASE = Low16bits(NEXT_LATCHES.REGS[BASER]);
		DATA OFFSET6 = sext((IR & ((1 << 6) - 1)), 6);
		DATA ADDRESS = BASE + OFFSET6;
		DATA RESULT = Low16bits(MEMORY[ADDRESS]);
		NEXT_LATCHES.REGS[DR] = To32bits(RESULT);
		setcc(RESULT);
		break;
	}

	case OPCODE_LEA:
	{
		uDATA DR = Low3bits(IR >> 9);
		DATA OFFSET9 = sext((IR & ((1 << 9) - 1)), 9);
		DATA PC = Low16bits(NEXT_LATCHES.PC);
		DATA RESULT = PC + OFFSET9;
		NEXT_LATCHES.REGS[DR] = To32bits(RESULT);
		break;
	}

	case OPCODE_NOT:
	{
		uDATA DR = Low3bits(IR >> 9);
		uDATA SR = Low3bits(IR >> 6);
		DATA OPERAND = Low16bits(NEXT_LATCHES.REGS[SR]);
		DATA RESULT = ~OPERAND;
		NEXT_LATCHES.REGS[DR] = To32bits(RESULT);
		setcc(RESULT);
		break;
	}

	case OPCODE_RTI:
	{
		break;
	}

	case OPCODE_ST:
	{
		uDATA SR = Low3bits(IR >> 9);
		DATA OFFSET9 = sext((IR & ((1 << 9) - 1)), 9);
		DATA PC = Low16bits(NEXT_LATCHES.PC);
		DATA ADDRESS = PC + OFFSET9;
		DATA RESULT = Low16bits(NEXT_LATCHES.REGS[SR]);
		MEMORY[ADDRESS] = To32bits(RESULT);
		break;
	}

	case OPCODE_STI:
	{
		uDATA SR = Low3bits(IR >> 9);
		DATA OFFSET9 = sext((IR & ((1 << 9) - 1)), 9);
		DATA PC = Low16bits(NEXT_LATCHES.PC);
		DATA ADDRESS = PC + OFFSET9;
		ADDRESS = Low16bits(MEMORY[ADDRESS]);
		DATA RESULT = Low16bits(NEXT_LATCHES.REGS[SR]);
		MEMORY[ADDRESS] = To32bits(RESULT);
		break;
	}

	case OPCODE_STR:
	{
		uDATA SR = Low3bits(IR >> 9);
		uDATA BASER = Low3bits(IR >> 6);
		DATA BASE = Low16bits(NEXT_LATCHES.REGS[BASER]);
		DATA OFFSET6 = sext((IR & ((1 << 6) - 1)), 6);
		DATA ADDRESS = BASE + OFFSET6;
		DATA RESULT = Low16bits(NEXT_LATCHES.REGS[SR]);
		MEMORY[ADDRESS] = To32bits(RESULT);
		break;
	}

	case OPCODE_TRAP:
	{
		uDATA VECTOR = (IR & 0xFF);
		if (VECTOR == 0x20) // trap GETC
		{
			DATA VALUE = getchar();
			NEXT_LATCHES.REGS[0] = To32bits(VALUE);
		}
		else if (VECTOR == 0x21) // trap OUT
		{
			DATA VALUE = Low16bits(NEXT_LATCHES.REGS[0]);
			printf("%c", (VALUE & 0xFF));
		}
		else if (VECTOR == 0x22) // trap PUTS
		{
			DATA ADDRESS = Low16bits(NEXT_LATCHES.REGS[0]);
			while (1)
			{
				DATA VALUE = Low16bits(MEMORY[ADDRESS]);
				if (VALUE == 0)
					break;
				printf("%c", (VALUE & 0xFF));
				ADDRESS = ADDRESS + 1;
			}
		}
		else if (VECTOR == 0x23) // trap IN
		{
			printf("Input a character>");
			DATA VALUE = getchar();
			NEXT_LATCHES.REGS[0] = To32bits(VALUE);
			printf("%c\n", (VALUE & 0xFF));
		}
		else if (VECTOR == 0x24) // trap PUTSP
		{
			DATA ADDRESS = Low16bits(NEXT_LATCHES.REGS[0]);
			while (1)
			{
				DATA VALUE = Low16bits(MEMORY[ADDRESS]);
				if (VALUE == 0)
					break;
				DATA FIRST = (VALUE & 0xFF);
				printf("%c", (FIRST & 0xFF));
				DATA SECOND = (VALUE >> 8);
				if (SECOND)
					printf("%c", (SECOND & 0xFF));
				ADDRESS = ADDRESS + 1;
			}
		}
		else if (VECTOR == 0x25) // trap HALT
		{
			NEXT_LATCHES.PC = 0;
			printf("\n----- Halting the processor -----\n");
		}
		NEXT_LATCHES.PC = 0;
		break;
	}

	case OPCODE_RESERVED:
	{
		break;
	}
	}
}