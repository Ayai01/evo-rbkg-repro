module FirStream #(
 parameter integer SIM_CONFIG=1, CURRENT_INTERFACE=0, DATA_WIDTH=16, COEFFICIENT_WIDTH=16,
 parameter integer ACCUMULATOR_WIDTH=32, MAC_LANES=2, TAP_COUNT=64,
 parameter integer FIFO_FRAMES=32, REPLAY_FRAMES=128,
 parameter integer P0=0,P1=0,P2=0,P3=0,P4=0,P5=0,P6=0,
 parameter integer P7=0,P8=0,P9=0,P10=0,P11=0,P12=0,P13=0
) (
 input logic clk, reset, step,
 input logic [41:0] schemes,
 input logic [29:0] bank_schemes,
 input logic [6:0] taps, fifo_depth,
 input logic [7:0] replay_depth,
 input logic [5:0] data_bits, coefficient_bits, accumulator_bits,
 input logic [2:0] lanes,
 input logic rounding_nearest, saturate, scrub,
 input logic [1:0] recovery_policy,
 input logic input_valid,
 input logic [31:0] input_frame,
 input logic signed [15:0] input0, input1, input2, input3,
 input logic coefficient_load,
 input logic [5:0] coefficient_index,
 input logic fault_valid,
 input logic [3:0] fault_domain,
 input logic [9:0] fault_word,
 input logic [7:0] fault_bit,
 input logic [1:0] fault_mode,
 input logic [191:0] fault_value,
 input logic [3:0] set_valid,
 input logic [63:0] set0, set1, set2, set3,
 input logic output_ready,
 output logic output_valid,
 output logic [31:0] output_frame,
 output logic signed [15:0] output0, output1, output2, output3,
 output logic halted,
 output logic [31:0] received_frames, accepted_frames, overflow_frames, recovery_cycles,
 output logic [31:0] replayed_frames, reloaded_coefficients, recovery_attempts,
 output logic [31:0] peak_backlog_frames, mac_operations, encoded_writes,
 output logic [31:0] service_cycles, encoded_write_bits,
 output logic [31:0] corrected_reads, due_events, flushed_frames,
 output logic [31:0] pending_frames,
 output logic fault_probe,
 output logic [63:0] probe_data,
 output logic [191:0] probe_encoded_out,
 output logic [6:0] probe_payload_bits,
 output logic probe_capture_written
);
 localparam integer HISTORY_CAPACITY=SIM_CONFIG!=0 ? 64 : TAP_COUNT;
 localparam integer FIFO_CAPACITY=SIM_CONFIG!=0 ? 64 : FIFO_FRAMES;
 localparam integer REPLAY_CAPACITY=SIM_CONFIG!=0 ? 128 : REPLAY_FRAMES;
 wire [6:0] effective_taps=SIM_CONFIG!=0 ? taps : 7'(TAP_COUNT);
 wire [6:0] effective_fifo_depth=fifo_depth;
 wire [7:0] effective_replay_depth=replay_depth;
 wire [5:0] effective_data_bits=SIM_CONFIG!=0 ? data_bits : 6'(DATA_WIDTH);
 wire [5:0] effective_coefficient_bits=SIM_CONFIG!=0 ? coefficient_bits : 6'(COEFFICIENT_WIDTH);
 wire [5:0] effective_accumulator_bits=SIM_CONFIG!=0 ? accumulator_bits : 6'(ACCUMULATOR_WIDTH);
 wire [2:0] effective_lanes=lanes;
 localparam integer IDLE=0, MAC=1, EMIT=2, CLEAR=3, COEFFICIENTS=4,
                    RESTORE=5, HALT=7;
 logic [191:0] history [0:4*HISTORY_CAPACITY-1];
 logic [191:0] coefficients [0:HISTORY_CAPACITY-1];
 logic [191:0] fifo [0:5*FIFO_CAPACITY-1];
 logic [191:0] replay [0:5*REPLAY_CAPACITY-1];
 logic [191:0] accumulators [0:3];
 logic [191:0] state [0:21];
 integer values [0:21];
 integer rd, wr, count, old_count, mode, phase, head, current_head;
 integer word, frame, writes, write_bits, corrections, due_count;
 logic [21:0] written_state;
 // Trusted testbench observations of the last functional edge. These are
 // neither fault targets nor part of the engineering implementation cost.
 logic [4*HISTORY_CAPACITY-1:0] current_history_written;
 logic [HISTORY_CAPACITY-1:0] current_coefficients_written;
 logic [5*FIFO_CAPACITY-1:0] current_fifo_written;
 logic [5*REPLAY_CAPACITY-1:0] current_replay_written;
 logic [3:0] current_accumulators_written;
 integer clear_index, cursor, target, anchor, last_commit;
 integer tap_index, index, raw, shift, fifo_slot, replay_slot, probe_scheme;
 logic [65:0] decoded, coeff_decoded, history_decoded, output_valid_decoded;
 logic bad_read, control_due, output_due, playing;
 logic signed [63:0] sums [0:3];
 logic signed [63:0] quantized;
 logic [63:0] transient [0:3];
 logic [191:0] probe_encoded;
 logic [65:0] probe_decoded;
 integer probe_index, probe_width;

 function automatic logic signed [15:0] fixed_coefficient(input integer address);
  begin
   fixed_coefficient=0;
   if (effective_taps==32) case (address)
    0: fixed_coefficient=16'sd27;
    1: fixed_coefficient=16'sd27;
    2: fixed_coefficient=16'sd22;
    3: fixed_coefficient=16'sd0;
    4: fixed_coefficient=-16'sd49;
    5: fixed_coefficient=-16'sd124;
    6: fixed_coefficient=-16'sd208;
    7: fixed_coefficient=-16'sd260;
    8: fixed_coefficient=-16'sd230;
    9: fixed_coefficient=-16'sd66;
    10: fixed_coefficient=16'sd258;
    11: fixed_coefficient=16'sd731;
    12: fixed_coefficient=16'sd1297;
    13: fixed_coefficient=16'sd1863;
    14: fixed_coefficient=16'sd2322;
    15: fixed_coefficient=16'sd2582;
    16: fixed_coefficient=16'sd2582;
    17: fixed_coefficient=16'sd2322;
    18: fixed_coefficient=16'sd1863;
    19: fixed_coefficient=16'sd1297;
    20: fixed_coefficient=16'sd731;
    21: fixed_coefficient=16'sd258;
    22: fixed_coefficient=-16'sd66;
    23: fixed_coefficient=-16'sd230;
    24: fixed_coefficient=-16'sd260;
    25: fixed_coefficient=-16'sd208;
    26: fixed_coefficient=-16'sd124;
    27: fixed_coefficient=-16'sd49;
    28: fixed_coefficient=16'sd0;
    29: fixed_coefficient=16'sd22;
    30: fixed_coefficient=16'sd27;
    31: fixed_coefficient=16'sd27;
    default: fixed_coefficient=0;
   endcase
   else if (effective_taps==64) case (address)
    0: fixed_coefficient=-16'sd2;
    1: fixed_coefficient=16'sd5;
    2: fixed_coefficient=16'sd12;
    3: fixed_coefficient=16'sd18;
    4: fixed_coefficient=16'sd21;
    5: fixed_coefficient=16'sd18;
    6: fixed_coefficient=16'sd8;
    7: fixed_coefficient=-16'sd10;
    8: fixed_coefficient=-16'sd33;
    9: fixed_coefficient=-16'sd56;
    10: fixed_coefficient=-16'sd69;
    11: fixed_coefficient=-16'sd65;
    12: fixed_coefficient=-16'sd37;
    13: fixed_coefficient=16'sd15;
    14: fixed_coefficient=16'sd80;
    15: fixed_coefficient=16'sd144;
    16: fixed_coefficient=16'sd185;
    17: fixed_coefficient=16'sd181;
    18: fixed_coefficient=16'sd119;
    19: fixed_coefficient=16'sd0;
    20: fixed_coefficient=-16'sd159;
    21: fixed_coefficient=-16'sd322;
    22: fixed_coefficient=-16'sd442;
    23: fixed_coefficient=-16'sd468;
    24: fixed_coefficient=-16'sd358;
    25: fixed_coefficient=-16'sd91;
    26: fixed_coefficient=16'sd325;
    27: fixed_coefficient=16'sd850;
    28: fixed_coefficient=16'sd1420;
    29: fixed_coefficient=16'sd1951;
    30: fixed_coefficient=16'sd2362;
    31: fixed_coefficient=16'sd2590;
    32: fixed_coefficient=16'sd2590;
    33: fixed_coefficient=16'sd2362;
    34: fixed_coefficient=16'sd1951;
    35: fixed_coefficient=16'sd1420;
    36: fixed_coefficient=16'sd850;
    37: fixed_coefficient=16'sd325;
    38: fixed_coefficient=-16'sd91;
    39: fixed_coefficient=-16'sd358;
    40: fixed_coefficient=-16'sd468;
    41: fixed_coefficient=-16'sd442;
    42: fixed_coefficient=-16'sd322;
    43: fixed_coefficient=-16'sd159;
    44: fixed_coefficient=16'sd0;
    45: fixed_coefficient=16'sd119;
    46: fixed_coefficient=16'sd181;
    47: fixed_coefficient=16'sd185;
    48: fixed_coefficient=16'sd144;
    49: fixed_coefficient=16'sd80;
    50: fixed_coefficient=16'sd15;
    51: fixed_coefficient=-16'sd37;
    52: fixed_coefficient=-16'sd65;
    53: fixed_coefficient=-16'sd69;
    54: fixed_coefficient=-16'sd56;
    55: fixed_coefficient=-16'sd33;
    56: fixed_coefficient=-16'sd10;
    57: fixed_coefficient=16'sd8;
    58: fixed_coefficient=16'sd18;
    59: fixed_coefficient=16'sd21;
    60: fixed_coefficient=16'sd18;
    61: fixed_coefficient=16'sd12;
    62: fixed_coefficient=16'sd5;
    63: fixed_coefficient=-16'sd2;
    default: fixed_coefficient=0;
   endcase
  end
 endfunction

 function automatic integer check_bits(input integer width);
  integer result;
  begin
   result=2;
   while ((1<<result)<width+result+1) result=result+1;
   check_bits=result;
  end
 endfunction
 function automatic logic [191:0] state_encode(input logic [63:0] data,
                                               input logic [2:0] protection,
                                               input integer width);
  logic [191:0] code;
  integer checks, hamming, data_index;
  begin
   code='0;
   checks=check_bits(width);
   hamming=width+checks;
   data_index=0;
   case (protection)
    0,3: begin
     for (integer bitno=0;bitno<64;bitno=bitno+1) if (bitno<width) code[bitno]=data[bitno];
     if (protection==3) code[width]=^code;
    end
    1: begin
     for (integer bitno=0;bitno<64;bitno=bitno+1) if (bitno<width) begin
      code[bitno]=data[bitno]; code[bitno+width]=data[bitno]; code[bitno+2*width]=data[bitno];
     end
    end
    2: begin
     for (integer pos=1;pos<=71;pos=pos+1) if (pos<=hamming && (pos & (pos-1))!=0) begin
      code[pos-1]=data[data_index]; data_index=data_index+1;
     end
     for (integer parity=0;parity<7;parity=parity+1) if (parity<checks) begin
      for (integer pos=1;pos<=71;pos=pos+1)
       if (pos<=hamming && (pos & (1<<parity))!=0 && pos!=(1<<parity))
        code[(1<<parity)-1]=code[(1<<parity)-1]^code[pos-1];
     end
     code[hamming]=^code;
    end
    default: code='0;
   endcase
   state_encode=code;
  end
 endfunction
 function automatic logic [65:0] state_decode(input logic [191:0] stored,
                                              input logic [2:0] protection,
                                              input integer width);
  logic [191:0] code;
  logic [63:0] data;
  logic corrected, due, parity;
  integer checks, hamming, syndrome, data_index;
  begin
   code=stored; data='0; corrected=0; due=0; syndrome=0;
   checks=check_bits(width); hamming=width+checks; data_index=0; parity=0;
   case (protection)
    0,3: begin
     for (integer bitno=0;bitno<64;bitno=bitno+1) if (bitno<width) data[bitno]=code[bitno];
     if (protection==3) begin
      for (integer bitno=0;bitno<65;bitno=bitno+1) if (bitno<=width) parity=parity^code[bitno];
      due=parity;
     end
    end
    1: begin
     for (integer bitno=0;bitno<64;bitno=bitno+1) if (bitno<width) begin
      data[bitno]=(code[bitno]&code[bitno+width])|(code[bitno]&code[bitno+2*width])|(code[bitno+width]&code[bitno+2*width]);
      corrected=corrected || code[bitno]!=code[bitno+width] || code[bitno]!=code[bitno+2*width];
     end
    end
    2: begin
     for (integer pos=1;pos<=71;pos=pos+1) if (pos<=hamming) begin
      if (code[pos-1]) syndrome=syndrome^pos;
      parity=parity^code[pos-1];
     end
     parity=parity^code[hamming];
     if (parity && syndrome!=0) begin
      if (syndrome<=hamming) begin code[syndrome-1]=~code[syndrome-1]; corrected=1; end
      else due=1;
     end else if (parity) corrected=1;
     else if (syndrome!=0) due=1;
     for (integer pos=1;pos<=71;pos=pos+1) if (pos<=hamming && (pos & (pos-1))!=0) begin
      data[data_index]=code[pos-1]; data_index=data_index+1;
     end
    end
    default: due=1;
   endcase
   state_decode={due,corrected,data};
  end
 endfunction

 function automatic integer reg_domain(input integer r);
  case (r)
   0,1,2: reg_domain=8;
   3,4: reg_domain=9;
   5: reg_domain=10;
   6: reg_domain=23;
   7,8,14: reg_domain=11;
   9,10,11,12,13,21: reg_domain=12;
   19,20: reg_domain=22;
   default: reg_domain=13;
  endcase
 endfunction
 function automatic integer reg_width(input integer r);
  case (r)
   0,1,4,14: reg_width=6;
   2,7,12: reg_width=7;
   3,13: reg_width=2;
   8: reg_width=4;
   20,21: reg_width=1;
   15,16,17,18: reg_width=16;
   default: reg_width=32;
  endcase
 endfunction
 function automatic integer hist_bank(input integer ch, tap);
  hist_bank=tap<16 ? ch : 14+ch;
 endfunction
 function automatic integer acc_bank(input integer ch);
  acc_bank=ch==0 ? 7 : 17+ch;
 endfunction
 function automatic integer replay_bank(input integer address);
  replay_bank=address<4*REPLAY_CAPACITY ? 6 : 21;
 endfunction
 function automatic logic [2:0] scheme(input integer d);
  if (d>=14 && d<24) scheme=bank_schemes[(d-14)*3 +: 3];
  else if (d<4 || d==6 || d==7 || d==11 || d==13) scheme=schemes[d*3 +: 3];
  else case (d)
   0:scheme=3'(P0); 1:scheme=3'(P1); 2:scheme=3'(P2); 3:scheme=3'(P3); 4:scheme=3'(P4); 5:scheme=3'(P5);
   6:scheme=3'(P6); 7:scheme=3'(P7); 8:scheme=3'(P8); 9:scheme=3'(P9); 10:scheme=3'(P10);
   11:scheme=3'(P11); 12:scheme=3'(P12); 13:scheme=3'(P13); default:scheme=0;
  endcase
 endfunction
 function automatic logic [191:0] encode(input logic [63:0] v, input integer d, w);
  encode=state_encode(64'(v),scheme(d),w);
 endfunction
 function automatic logic [65:0] read_word(input logic [191:0] v, input integer d, w);
  read_word=state_decode(v,scheme(d),w);
 endfunction
 function automatic logic [191:0] fault_result(input logic [191:0] v, input integer d, w);
  if (CURRENT_INTERFACE!=0 && fault_mode==1) fault_result=fault_value;
  else if (CURRENT_INTERFACE!=0 && fault_mode==2) fault_result=encode(fault_value[63:0],d,w);
  else fault_result=v^(192'd1<<fault_bit);
 endfunction
 task automatic write_history(input integer at, w, input logic [63:0] v);
  history[at]<=encode(v,hist_bank(at/HISTORY_CAPACITY,at%HISTORY_CAPACITY),w);
  if (CURRENT_INTERFACE!=0) current_history_written[at]=1;
 endtask
 task automatic write_coefficient(input integer at, w, input logic [63:0] v);
  coefficients[at]<=encode(v,4,w);
  if (CURRENT_INTERFACE!=0) current_coefficients_written[at]=1;
 endtask
 task automatic write_fifo(input integer at, w, input logic [63:0] v);
  fifo[at]<=encode(v,5,w);
  if (CURRENT_INTERFACE!=0) current_fifo_written[at]=1;
 endtask
 task automatic write_replay(input integer at, w, input logic [63:0] v);
  replay[at]<=encode(v,replay_bank(at),w);
  if (CURRENT_INTERFACE!=0) current_replay_written[at]=1;
 endtask
 task automatic write_accumulator(input integer at, w, input logic [63:0] v);
  accumulators[at]<=encode(v,acc_bank(at),w);
  if (CURRENT_INTERFACE!=0) current_accumulators_written[at]=1;
 endtask
 function automatic logic signed [63:0] extend_word(input logic [63:0] v, input integer w);
  extend_word=$signed(v << (64-w)) >>> (64-w);
 endfunction
 function automatic logic signed [63:0] requantize(input logic signed [63:0] v, input integer amount);
  begin
   if (amount<=0) requantize=v<<(-amount);
   else if (rounding_nearest) requantize=v<0 ? -((-v+(64'sd1<<(amount-1)))>>>amount) : (v+(64'sd1<<(amount-1)))>>>amount;
   else requantize=v>>>amount;
  end
 endfunction
 function automatic logic signed [63:0] quantize_sample(input logic signed [63:0] v);
  logic signed [63:0] result, bound;
  begin
   result=requantize(v,16-int'(effective_data_bits));
   bound=64'sd1<<(int'(effective_data_bits)-1);
   quantize_sample=result>=bound ? bound-1 : result< -bound ? -bound : result;
  end
 endfunction
 function automatic logic signed [63:0] multiply(input logic signed [63:0] a, b);
  logic signed [DATA_WIDTH-1:0] narrow_a;
  logic signed [COEFFICIENT_WIDTH-1:0] narrow_b;
  logic signed [DATA_WIDTH+COEFFICIENT_WIDTH-1:0] product;
  begin
   narrow_a=DATA_WIDTH'(a); narrow_b=COEFFICIENT_WIDTH'(b);
   product=narrow_a*narrow_b;
   multiply=SIM_CONFIG!=0 ? a*b : 64'(product);
  end
 endfunction
 function automatic logic signed [63:0] accumulate(input logic signed [63:0] a, b);
  logic signed [ACCUMULATOR_WIDTH-1:0] narrowed;
  begin
   narrowed=ACCUMULATOR_WIDTH'(a+b);
   accumulate=SIM_CONFIG!=0 ? a+b : 64'(narrowed);
  end
 endfunction
 task automatic count_write(input integer d, w);
  integer checks, physical;
  begin
   checks=2;
   while ((1<<checks)<w+checks+1) checks=checks+1;
   case (scheme(d))
    1: physical=3*w;
    2: physical=w+checks+1;
    3: physical=w+1;
    default: physical=w;
   endcase
   writes=writes+1;
   write_bits=write_bits+physical;
  end
 endtask
 task automatic write_state(input integer r, input logic [31:0] v);
  state[r]<=encode(64'(v),reg_domain(r),reg_width(r));
  if (!written_state[r]) count_write(reg_domain(r),reg_width(r));
  written_state[r]=1;
 endtask
 task automatic recover(input integer end_frame);
  due_count=due_count+1;
  recovery_attempts<=recovery_attempts+1;
  for (integer r=15;r<=20;r=r+1) write_state(r,0);
  if (recovery_policy==0 || values[13]>=2 || control_due) begin
   write_state(8,HALT);
  end else begin
   write_state(13,32'(values[13]+1));
   write_state(10,32'(end_frame));
   write_state(12,0);
   write_state(8,CLEAR);
   if (recovery_policy==1) begin
    flushed_frames<=flushed_frames+32'(count);
    rd=wr;
    count=0;
    write_state(21,0);
   end else begin
    last_commit=values[6];
    anchor=last_commit<0 ? 0 : (last_commit>=int'(effective_taps)-1 ? last_commit-int'(effective_taps)+1 : 0);
    write_state(11,32'(anchor));
    write_state(9,32'(anchor));
    write_state(21,1);
   end
  end
 endtask

 always_comb begin
  output_valid_decoded=read_word(state[20],22,1);
  output_due=output_valid_decoded[65];
  for (integer j=15;j<20;j=j+1)
   output_due=output_due || (read_word(state[j],reg_domain(j),reg_width(j)) >> 65)!=0;
  output_valid=output_valid_decoded[0] && !output_due && 32'(read_word(state[8],11,4))==EMIT;
  output_frame=32'(read_word(state[19],22,32));
  output0=16'(read_word(state[15],13,16));
  output1=16'(read_word(state[16],13,16));
  output2=16'(read_word(state[17],13,16));
  output3=16'(read_word(state[18],13,16));
  halted=32'(read_word(state[8],11,4))==HALT;
  pending_frames=32'(read_word(state[2],8,7));
  probe_encoded='0; probe_width=32; probe_index=-1;
  case (fault_domain)
   0,1,2,3: begin
    if (int'(fault_word)<int'(effective_taps)) probe_encoded=history[int'(fault_domain)*HISTORY_CAPACITY+int'(fault_word)];
    probe_width=int'(effective_data_bits);
   end
   4: begin
    if (int'(fault_word)<int'(effective_taps)) probe_encoded=coefficients[int'(fault_word)];
    probe_width=int'(effective_coefficient_bits);
   end
   5: begin
    probe_index=int'(fault_word)<4*int'(effective_fifo_depth) ? (int'(fault_word)/int'(effective_fifo_depth))*FIFO_CAPACITY+int'(fault_word)%int'(effective_fifo_depth) : 4*FIFO_CAPACITY+int'(fault_word)-4*int'(effective_fifo_depth);
    if (probe_index<5*FIFO_CAPACITY) probe_encoded=fifo[probe_index];
    probe_width=int'(fault_word)<4*int'(effective_fifo_depth) ? 16 : 32;
   end
   6: begin
    probe_index=int'(fault_word)<4*int'(effective_replay_depth) ? (int'(fault_word)/int'(effective_replay_depth))*REPLAY_CAPACITY+int'(fault_word)%int'(effective_replay_depth) : 4*REPLAY_CAPACITY+int'(fault_word)-4*int'(effective_replay_depth);
    if (probe_index<5*REPLAY_CAPACITY) probe_encoded=replay[probe_index];
    probe_width=int'(fault_word)<4*int'(effective_replay_depth) ? 16 : 32;
   end
   7: begin
    if (fault_word<4) probe_encoded=accumulators[int'(fault_word)];
    probe_width=int'(effective_accumulator_bits);
   end
   default: begin
    case (fault_domain)
     8: if (fault_word<3) probe_index=int'(fault_word);
     9: if (fault_word<2) probe_index=3+int'(fault_word);
     10: if (fault_word<2) probe_index=5+int'(fault_word);
     11: case (fault_word) 0:probe_index=7; 1:probe_index=8; 2:probe_index=14; default:probe_index=-1; endcase
     12: case (fault_word) 0:probe_index=9; 1:probe_index=10; 2:probe_index=11; 3:probe_index=12; 4:probe_index=13; 5:probe_index=21; default:probe_index=-1; endcase
     13: if (fault_word<6) probe_index=15+int'(fault_word);
     default: probe_index=-1;
    endcase
    if (probe_index>=0) begin probe_encoded=state[probe_index]; probe_width=reg_width(probe_index); end
   end
  endcase
  fault_probe=probe_encoded[fault_bit];
  probe_scheme=int'(fault_domain);
  if (fault_domain<4) probe_scheme=hist_bank(int'(fault_domain),int'(fault_word));
  else if (fault_domain==7) probe_scheme=acc_bank(int'(fault_word));
  else if (fault_domain==6) probe_scheme=replay_bank(probe_index);
  else if (fault_domain>=8 && probe_index>=0) probe_scheme=reg_domain(probe_index);
  probe_decoded=read_word(probe_encoded,probe_scheme,probe_width);
  probe_data=probe_decoded[63:0];
  probe_encoded_out=probe_encoded;
  probe_payload_bits=7'(probe_width);
  probe_capture_written=0;
  if (CURRENT_INTERFACE!=0) case (fault_domain)
   0,1,2,3: if (int'(fault_word)<int'(effective_taps)) probe_capture_written=current_history_written[int'(fault_domain)*HISTORY_CAPACITY+int'(fault_word)];
   4: if (int'(fault_word)<int'(effective_taps)) probe_capture_written=current_coefficients_written[int'(fault_word)];
   5: if (probe_index>=0 && probe_index<5*FIFO_CAPACITY) probe_capture_written=current_fifo_written[probe_index];
   6: if (probe_index>=0 && probe_index<5*REPLAY_CAPACITY) probe_capture_written=current_replay_written[probe_index];
   7: if (fault_word<4) probe_capture_written=current_accumulators_written[int'(fault_word)];
   default: if (probe_index>=0 && probe_index<22) probe_capture_written=written_state[probe_index];
  endcase
 end

 always_ff @(posedge clk) begin
  if (reset) begin
   current_history_written='0;current_coefficients_written='0;current_fifo_written='0;
   current_replay_written='0;current_accumulators_written='0;written_state='0;
   for (integer j=0;j<4*HISTORY_CAPACITY;j=j+1) history[j]<=encode(0,hist_bank(j/HISTORY_CAPACITY,j%HISTORY_CAPACITY),int'(effective_data_bits));
   for (integer j=0;j<5*FIFO_CAPACITY;j=j+1) fifo[j]<=encode(0,5,j<4*FIFO_CAPACITY ? 16 : 32);
   for (integer j=0;j<5*REPLAY_CAPACITY;j=j+1) replay[j]<=encode(0,replay_bank(j),j<4*REPLAY_CAPACITY ? 16 : 32);
   for (integer j=0;j<4;j=j+1) accumulators[j]<=encode(0,acc_bank(j),int'(effective_accumulator_bits));
   for (integer j=0;j<22;j=j+1) state[j]<=encode(j==6 ? 64'hffffffff : 64'd0,reg_domain(j),reg_width(j));
   received_frames<=0; accepted_frames<=0; overflow_frames<=0; recovery_cycles<=0;
   replayed_frames<=0; reloaded_coefficients<=0; recovery_attempts<=0;
   peak_backlog_frames<=0; mac_operations<=0; encoded_writes<=0; service_cycles<=0; encoded_write_bits<=0;
   corrected_reads<=0; due_events<=0; flushed_frames<=0;
  end else if (coefficient_load) begin
   if (int'(coefficient_index)<HISTORY_CAPACITY)
    write_coefficient(int'(coefficient_index),int'(effective_coefficient_bits),requantize(64'($signed(fixed_coefficient(int'(coefficient_index)))),16-int'(effective_coefficient_bits)));
  end else if (fault_valid) begin
   case (fault_domain)
    0,1,2,3: if (int'(fault_word)<int'(effective_taps)) history[int'(fault_domain)*HISTORY_CAPACITY+int'(fault_word)]<=fault_result(history[int'(fault_domain)*HISTORY_CAPACITY+int'(fault_word)],hist_bank(int'(fault_domain),int'(fault_word)),int'(effective_data_bits));
    4: if (int'(fault_word)<int'(effective_taps)) coefficients[int'(fault_word)]<=fault_result(coefficients[int'(fault_word)],4,int'(effective_coefficient_bits));
    5: begin
     word=int'(fault_word)<4*int'(effective_fifo_depth) ? (int'(fault_word)/int'(effective_fifo_depth))*FIFO_CAPACITY+int'(fault_word)%int'(effective_fifo_depth) : 4*FIFO_CAPACITY+int'(fault_word)-4*int'(effective_fifo_depth);
     if (word<5*FIFO_CAPACITY) fifo[word]<=fault_result(fifo[word],5,int'(fault_word)<4*int'(effective_fifo_depth) ? 16 : 32);
    end
    6: begin
     word=int'(fault_word)<4*int'(effective_replay_depth) ? (int'(fault_word)/int'(effective_replay_depth))*REPLAY_CAPACITY+int'(fault_word)%int'(effective_replay_depth) : 4*REPLAY_CAPACITY+int'(fault_word)-4*int'(effective_replay_depth);
     if (word<5*REPLAY_CAPACITY) replay[word]<=fault_result(replay[word],replay_bank(word),int'(fault_word)<4*int'(effective_replay_depth) ? 16 : 32);
    end
    7: if (fault_word<4) accumulators[int'(fault_word)]<=fault_result(accumulators[int'(fault_word)],acc_bank(int'(fault_word)),int'(effective_accumulator_bits));
    default: begin
     word=-1;
     case (fault_domain)
      8: if (fault_word<3) word=int'(fault_word);
      9: if (fault_word<2) word=3+int'(fault_word);
      10: if (fault_word<2) word=5+int'(fault_word);
      11: case (fault_word) 0:word=7; 1:word=8; 2:word=14; default:word=-1; endcase
      12: case (fault_word) 0:word=9; 1:word=10; 2:word=11; 3:word=12; 4:word=13; 5:word=21; default:word=-1; endcase
      13: if (fault_word<6) word=15+int'(fault_word);
      default:word=-1;
     endcase
     if (word>=0) state[word]<=fault_result(state[word],reg_domain(word),reg_width(word));
    end
   endcase
  end else if (step) begin
   current_history_written='0;current_coefficients_written='0;current_fifo_written='0;
   current_replay_written='0;current_accumulators_written='0;
   writes=0; write_bits=0; written_state='0; corrections=0; due_count=0; control_due=0;
   for (integer j=0;j<22;j=j+1) begin
    decoded=read_word(state[j],reg_domain(j),reg_width(j));
    values[j]=32'(decoded);
    begin
     if (j<15 || j==21) control_due=control_due || decoded[65];
     if (decoded[64]) begin
      corrections=corrections+1;
      if (scrub || scheme(reg_domain(j))==1) write_state(j,decoded[31:0]);
     end
    end
   end
   rd=values[0]; wr=values[1]; count=values[2]; old_count=count;
   phase=values[3]; head=values[4]; mode=values[8];
   if ((mode!=IDLE && mode!=HALT) || (mode==IDLE && (old_count>0 || values[21]!=0)))
    service_cycles<=service_cycles+1;
   transient[0]=set0; transient[1]=set1; transient[2]=set2; transient[3]=set3;
   if (rd>=int'(effective_fifo_depth) || wr>=int'(effective_fifo_depth) || count>int'(effective_fifo_depth) || head>=int'(effective_taps) || phase>3 || mode>7) begin
    control_due=1;
    rd=0; wr=0; count=0;
   end
   if ((values[21]!=0 || mode==RESTORE) && (values[9]<0 || values[10]<0)) control_due=1;
   if (input_valid) begin
    received_frames<=received_frames+1;
    replay_slot=int'(input_frame%32'(effective_replay_depth));
    write_replay(replay_slot,16,64'($signed(input0)));
    write_replay(REPLAY_CAPACITY+replay_slot,16,64'($signed(input1)));
    write_replay(2*REPLAY_CAPACITY+replay_slot,16,64'($signed(input2)));
    write_replay(3*REPLAY_CAPACITY+replay_slot,16,64'($signed(input3)));
    write_replay(4*REPLAY_CAPACITY+replay_slot,32,64'(input_frame));
    for (integer ch=0;ch<4;ch=ch+1) count_write(6,16);
    count_write(21,32);
    if (count<int'(effective_fifo_depth) && !control_due) begin
     accepted_frames<=accepted_frames+1;
     write_fifo(wr,16,64'($signed(input0)));
     write_fifo(FIFO_CAPACITY+wr,16,64'($signed(input1)));
     write_fifo(2*FIFO_CAPACITY+wr,16,64'($signed(input2)));
     write_fifo(3*FIFO_CAPACITY+wr,16,64'($signed(input3)));
     write_fifo(4*FIFO_CAPACITY+wr,32,64'(input_frame));
     for (integer ch=0;ch<4;ch=ch+1) count_write(5,16);
     count_write(5,32);
     wr=(wr+1)%int'(effective_fifo_depth);
     count=count+1;
    end else overflow_frames<=overflow_frames+1;
   end
   if (control_due && mode!=HALT) recover(values[5]);
   else case (mode)
    IDLE: begin
     playing=values[21]!=0;
     if (playing || old_count>0) begin
      cursor=playing ? values[9] : 0;
      fifo_slot=rd;
      replay_slot=cursor%int'(effective_replay_depth);
      decoded=playing ? read_word(replay[4*REPLAY_CAPACITY+replay_slot],21,32) : read_word(fifo[4*FIFO_CAPACITY+fifo_slot],5,32);
      frame=32'(decoded);
      bad_read=decoded[65] || (playing && frame!=cursor);
      corrections=corrections+int'(decoded[64]);
      if (decoded[64] && (scrub || scheme(playing ? 21 : 5)==1)
          && !(playing && input_valid && input_frame%32'(effective_replay_depth)==32'(replay_slot))) begin
       if (playing) write_replay(4*REPLAY_CAPACITY+replay_slot,32,decoded[63:0]);
       else write_fifo(4*FIFO_CAPACITY+fifo_slot,32,decoded[63:0]);
       count_write(playing ? 21 : 5,32);
      end
      for (integer ch=0;ch<4;ch=ch+1) begin
       history_decoded=playing ? read_word(replay[ch*REPLAY_CAPACITY+replay_slot],6,16) : read_word(fifo[ch*FIFO_CAPACITY+fifo_slot],5,16);
       bad_read=bad_read || history_decoded[65];
       raw=int'(quantize_sample(extend_word(history_decoded[63:0],16)));
       sums[ch]=64'(raw);
       corrections=corrections+int'(history_decoded[64]);
       if (history_decoded[64] && (scrub || scheme(playing ? 6 : 5)==1)
           && !(playing && input_valid && input_frame%32'(effective_replay_depth)==32'(replay_slot))) begin
        if (playing) write_replay(ch*REPLAY_CAPACITY+replay_slot,16,history_decoded[63:0]);
        else write_fifo(ch*FIFO_CAPACITY+fifo_slot,16,history_decoded[63:0]);
        count_write(playing ? 6 : 5,16);
       end
      end
      if (!playing) begin rd=(rd+1)%int'(effective_fifo_depth); count=count-1; end
      if (bad_read) recover(playing ? values[10] : frame);
      else begin
       for (integer ch=0;ch<4;ch=ch+1) begin
        write_history(ch*HISTORY_CAPACITY+head,int'(effective_data_bits),64'(sums[ch]));
        write_accumulator(ch,int'(effective_accumulator_bits),0);
        count_write(hist_bank(ch,head),int'(effective_data_bits));
        count_write(acc_bank(ch),int'(effective_accumulator_bits));
       end
       write_state(5,32'(frame));
       write_state(14,32'(head));
       write_state(4,32'((head+1)%int'(effective_taps)));
       write_state(3,32'((phase+1)%4));
       write_state(7,0);
       if (phase==3) write_state(8,MAC);
       if (playing) begin
        replayed_frames<=replayed_frames+1;
        write_state(9,32'(cursor+1));
        if (cursor>=values[10]) write_state(21,0);
       end
      end
     end
    end
    MAC: begin
     bad_read=0;
     tap_index=values[7]; current_head=values[14];
     if (tap_index>=int'(effective_taps) || current_head>=int'(effective_taps)) bad_read=1;
     for (integer ch=0;ch<4;ch=ch+1) begin
      decoded=read_word(accumulators[ch],acc_bank(ch),int'(effective_accumulator_bits));
      bad_read=bad_read || decoded[65];
      sums[ch]=64'(extend_word(decoded[63:0],int'(effective_accumulator_bits)));
      corrections=corrections+int'(decoded[64]);
     end
     for (integer pe=0;pe<4;pe=pe+1) if (pe<int'(effective_lanes) && tap_index+pe<int'(effective_taps)) begin
      index=(current_head+int'(effective_taps)-tap_index-pe)%int'(effective_taps);
      coeff_decoded=read_word(coefficients[tap_index+pe],4,int'(effective_coefficient_bits));
      bad_read=bad_read || coeff_decoded[65];
      corrections=corrections+int'(coeff_decoded[64]);
      if (coeff_decoded[64] && (scrub || scheme(4)==1)) begin
       write_coefficient(tap_index+pe,int'(effective_coefficient_bits),coeff_decoded[63:0]); count_write(4,int'(effective_coefficient_bits));
      end
      for (integer ch=0;ch<4;ch=ch+1) begin
       history_decoded=read_word(history[ch*HISTORY_CAPACITY+index],hist_bank(ch,index),int'(effective_data_bits));
       bad_read=bad_read || history_decoded[65];
       corrections=corrections+int'(history_decoded[64]);
       sums[ch]=accumulate(sums[ch],multiply(extend_word(history_decoded[63:0],int'(effective_data_bits)),extend_word(coeff_decoded[63:0],int'(effective_coefficient_bits))));
       if (history_decoded[64] && (scrub || scheme(hist_bank(ch,index))==1)) begin
        write_history(ch*HISTORY_CAPACITY+index,int'(effective_data_bits),history_decoded[63:0]); count_write(hist_bank(ch,index),int'(effective_data_bits));
       end
      end
     end
     if (bad_read) recover(values[5]);
     else begin
      for (integer ch=0;ch<4;ch=ch+1) begin
       if (set_valid[ch]) sums[ch]=sums[ch] ^ 64'(transient[ch]);
       write_accumulator(ch,int'(effective_accumulator_bits),64'(sums[ch]));
       count_write(acc_bank(ch),int'(effective_accumulator_bits));
      end
      mac_operations<=mac_operations+4*32'(effective_lanes);
      write_state(7,32'(tap_index+int'(effective_lanes)));
      if (tap_index+int'(effective_lanes)>=int'(effective_taps)) write_state(8,EMIT);
     end
    end
    EMIT: begin
     if (output_due) recover(values[5]);
     else if (values[20]!=0) begin
      if (output_ready) begin
       write_state(6,32'(values[5]));
       write_state(20,0); write_state(8,IDLE);
      end
     end else begin
      shift=14-(16-int'(effective_data_bits))-(16-int'(effective_coefficient_bits));
      bad_read=0;
      for (integer ch=0;ch<4;ch=ch+1) begin
       decoded=read_word(accumulators[ch],acc_bank(ch),int'(effective_accumulator_bits));
       bad_read=bad_read || decoded[65];
       quantized=64'(extend_word(decoded[63:0],int'(effective_accumulator_bits)));
       quantized=requantize(quantized,shift);
       if (saturate) begin
        if (quantized>32767) quantized=32767;
        if (quantized< -32768) quantized= -32768;
       end
       sums[ch]=quantized;
      end
      if (bad_read) recover(values[5]);
      else begin
       for (integer ch=0;ch<4;ch=ch+1) write_state(15+ch,32'(sums[ch]));
       write_state(19,32'(values[5])); write_state(20,1);
      end
     end
    end
    CLEAR: begin
     recovery_cycles<=recovery_cycles+1;
     clear_index=values[12];
     if (clear_index>=int'(effective_taps)) write_state(8,HALT);
     else begin
      for (integer ch=0;ch<4;ch=ch+1) begin
       write_history(ch*HISTORY_CAPACITY+clear_index,int'(effective_data_bits),0);
       count_write(hist_bank(ch,clear_index),int'(effective_data_bits));
      end
      write_state(12,32'(clear_index+1));
      if (clear_index+1>=int'(effective_taps)) begin
       write_state(12,0); write_state(4,0); write_state(3,0); write_state(8,COEFFICIENTS);
      end
     end
    end
    COEFFICIENTS: begin
     recovery_cycles<=recovery_cycles+1;
     clear_index=values[12];
     if (clear_index>=int'(effective_taps)) write_state(8,HALT);
     else begin
      write_coefficient(clear_index,int'(effective_coefficient_bits),requantize(64'($signed(fixed_coefficient(clear_index))),16-int'(effective_coefficient_bits)));
      count_write(4,int'(effective_coefficient_bits));
      reloaded_coefficients<=reloaded_coefficients+1;
      write_state(12,32'(clear_index+1));
      if (clear_index+1>=int'(effective_taps)) begin
       write_state(8,recovery_policy==1 || values[6]<0 ? IDLE : RESTORE);
       if (recovery_policy==1) write_state(3,32'((values[5]+1)%4));
      end
     end
    end
    RESTORE: begin
     recovery_cycles<=recovery_cycles+1;
     cursor=values[9];
     replay_slot=cursor%int'(effective_replay_depth);
     decoded=read_word(replay[4*REPLAY_CAPACITY+replay_slot],21,32);
     bad_read=decoded[65] || 32'(decoded)!=32'(cursor);
     corrections=corrections+int'(decoded[64]);
     if (decoded[64] && (scrub || scheme(21)==1)
         && !(input_valid && input_frame%32'(effective_replay_depth)==32'(replay_slot))) begin
      write_replay(4*REPLAY_CAPACITY+replay_slot,32,decoded[63:0]); count_write(21,32);
     end
     for (integer ch=0;ch<4;ch=ch+1) begin
      history_decoded=read_word(replay[ch*REPLAY_CAPACITY+replay_slot],6,16);
      bad_read=bad_read || history_decoded[65];
      sums[ch]=quantize_sample(extend_word(history_decoded[63:0],16));
      corrections=corrections+int'(history_decoded[64]);
      if (history_decoded[64] && (scrub || scheme(6)==1)
          && !(input_valid && input_frame%32'(effective_replay_depth)==32'(replay_slot))) begin
       write_replay(ch*REPLAY_CAPACITY+replay_slot,16,history_decoded[63:0]); count_write(6,16);
      end
     end
     if (bad_read) write_state(8,HALT);
     else begin
      for (integer ch=0;ch<4;ch=ch+1) begin
       write_history(ch*HISTORY_CAPACITY+head,int'(effective_data_bits),64'(sums[ch]));
       count_write(hist_bank(ch,head),int'(effective_data_bits));
      end
      write_state(4,32'((head+1)%int'(effective_taps)));
      write_state(3,32'((cursor+1)%4));
      write_state(9,32'(cursor+1));
      replayed_frames<=replayed_frames+1;
      if (cursor>=values[6]) write_state(8,IDLE);
     end
    end
    HALT: begin end
    default: recover(values[5]);
   endcase
   if (count>int'(peak_backlog_frames)) peak_backlog_frames<=32'(count);
   write_state(0,32'(rd)); write_state(1,32'(wr)); write_state(2,32'(count));
   encoded_writes<=encoded_writes+32'(writes);
   encoded_write_bits<=encoded_write_bits+32'(write_bits);
   corrected_reads<=corrected_reads+32'(corrections);
   due_events<=due_events+32'(due_count);
  end
 end
endmodule
