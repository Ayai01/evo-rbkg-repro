#include "VFirStream.h"
#include "verilated.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>
#include <set>
#include "../../framework/rtl/JointCapture.h"
#include "../../framework/rtl/ClockedInput.h"

static const int sine64[64] = {
 0,1176,2341,3483,4592,5657,6667,7613,8485,9276,9978,10583,11087,11483,11769,11942,
 12000,11942,11769,11483,11087,10583,9978,9276,8485,7613,6667,5657,4592,3483,2341,1176,
 0,-1176,-2341,-3483,-4592,-5657,-6667,-7613,-8485,-9276,-9978,-10583,-11087,-11483,-11769,-11942,
 -12000,-11942,-11769,-11483,-11087,-10583,-9978,-9276,-8485,-7613,-6667,-5657,-4592,-3483,-2341,-1176
};
static int floor_div(int value, int divisor) { return value>=0 ? value/divisor : -((-value+divisor-1)/divisor); }
static int sample(uint32_t frame, int channel, uint32_t seed) {
 uint32_t noise=(frame*1103515245u+uint32_t(channel)*12345u+seed*69069u)&65535u;
 uint32_t phase=(seed>>8)&63u, profile=(seed>>16)&3u;
 int value=sine64[(frame*uint32_t(channel+1)+uint32_t(channel)*7u+phase)%64u]
  +floor_div(sine64[(frame*13u+uint32_t(channel)*3u)%64u],4)+int(noise%601u)-300;
 if(profile==1) value=floor_div(value,4)+(frame/16u%2u ? 24000 : -24000);
 else if(profile==2) value=floor_div(value,2)+(frame%13u==seed%13u ? 30000 : 0);
 else if(profile==3) value=floor_div(value,8)+(frame/16u%2u ? 32767 : -32768);
 return std::max(-32768,std::min(32767,value));
}
struct Event { uint32_t cycle,domain,word,bit,response=0; };
static void tick(VFirStream& top) {
 top.clk=0; top.eval(); top.clk=1; top.eval(); top.clk=0; top.eval();
}

int main(int argc,char** argv) {
 Verilated::commandArgs(argc,argv);
 arh::JointCapture captures;arh::ClockedInput inputs;
 std::string line;
 while(std::getline(std::cin,line)) {
  try {
   // The bounded million-event schedule needs up to 19 ASCII bytes per event.
   if(line.size()>32*1024*1024) throw std::runtime_error("FIR request exceeds 32MiB");
   std::vector<int64_t> args;
   std::stringstream stream(line); std::string token;
   if(!std::getline(stream,token,','))throw std::runtime_error("empty FIR request");
   const std::string command=token;
   if(command=="C"||command=="S"||command=="I"){
    std::vector<std::string> fields{command};while(std::getline(stream,token,','))fields.push_back(token);
    if(command=="C"){captures.load_current(fields);std::cout<<"{\"classes\":"<<captures.size()<<"}"<<std::endl;}
    if(command=="S"){captures.styles(fields,14);std::cout<<"{\"style_words\":14}"<<std::endl;}
    if(command=="I"){inputs.load(fields,14*1024,63);std::cout<<"{\"input_configured\":true}"<<std::endl;}
    continue;
   }
   const bool joint=command=="J";
   if(!joint&&command!="T")throw std::runtime_error("unknown FIR request");
   while(std::getline(stream,token,',')) {
    size_t end=0; auto value=std::stoll(token,&end);
    if(end!=token.size()) throw std::runtime_error("noninteger FIR request");
    args.push_back(value);
   }
   if(args.size()<32) throw std::runtime_error("short FIR request");
   // seed, eleven implementation choices, fourteen protections, window and event count.
   uint32_t seed=uint32_t(args[0]);
   int taps=int(args[1]), data=int(args[2]), coefficient=int(args[3]), accumulator=int(args[4]);
   int frames=int(args[26]), period=int(args[27]), drain=int(args[28]), event_count=int(args[29]);
   int pulse_cycles=int(args[30]);
   if((taps!=32 && taps!=64) || data<8 || data>16 || coefficient<8 || coefficient>16
      || accumulator<24 || accumulator>40 || frames<4 || frames>512 || period<1 || period>1024
      || drain<1 || drain>1000000 || event_count<0 || event_count>1048576
      || pulse_cycles<1 || pulse_cycles>8
      || args.size()!=32+size_t(event_count)*(joint?5:4)) throw std::runtime_error("invalid FIR dimensions");
   auto context=std::make_unique<VerilatedContext>();
   VFirStream top(context.get());
   top.taps=uint8_t(taps); top.data_bits=uint8_t(data); top.coefficient_bits=uint8_t(coefficient);
   top.accumulator_bits=uint8_t(accumulator); top.rounding_nearest=uint8_t(args[5]); top.saturate=uint8_t(args[6]);
   top.fifo_depth=uint8_t(args[7]); top.replay_depth=uint8_t(args[8]); top.lanes=uint8_t(args[9]);
   top.recovery_policy=uint8_t(args[10]); top.scrub=uint8_t(args[11]);
   if((top.fifo_depth!=16 && top.fifo_depth!=32 && top.fifo_depth!=64)
      || (top.replay_depth!=32 && top.replay_depth!=64 && top.replay_depth!=128)
      || (top.lanes!=1 && top.lanes!=2 && top.lanes!=4)) throw std::runtime_error("invalid FIR implementation");
   uint64_t protections=0;
   for(int domain=0;domain<14;domain++) {
    if(args[12+domain]<0 || args[12+domain]>3) throw std::runtime_error("invalid FIR protection");
    protections|=uint64_t(args[12+domain])<<(domain*3);
   }
   if(args[31]<0 || args[31]>= (1LL<<30)) throw std::runtime_error("invalid FIR bank encoding");
   top.bank_schemes=uint32_t(args[31]); top.schemes=protections; top.reset=1; top.step=0; top.input_valid=0; top.output_ready=1;
   top.fault_valid=0;top.fault_mode=0;for(unsigned j=0;j<6;++j)top.fault_value[j]=0;
   top.coefficient_load=0; top.set_valid=0; top.set0=0; top.set1=0; top.set2=0; top.set3=0;
   tick(top); top.reset=0;
   // This immutable source is the actual fixed-coefficient ROM reload interface.
   for(int n=0;n<taps;n++) {
    top.coefficient_load=1; top.coefficient_index=uint8_t(n); tick(top);
   }
   top.coefficient_load=0;
   std::vector<Event> events;
   std::set<std::array<unsigned,4>> seen;captures.reset();inputs.reset();
   for(int n=0;n<event_count;n++) {
    size_t offset=32+size_t(n)*(joint?5:4);
    for(int j=0;j<4;j++) if(args[offset+j]<0) throw std::runtime_error("negative FIR fault field");
    Event event{uint32_t(args[offset]),uint32_t(args[offset+1]),uint32_t(args[offset+2]),uint32_t(args[offset+3])};
    if(joint)event.response=uint32_t(args[offset+4]);
    if(event.domain>(joint?13u:14u) || event.word>=640 || event.bit>=(joint?192u:128u)
       ||(joint&&(event.response>=captures.size()||!seen.insert({event.cycle,event.domain,event.word,event.bit}).second)))throw std::runtime_error("invalid FIR fault address");
    events.push_back(event);
   }
   std::stable_sort(events.begin(),events.end(),[](const Event&a,const Event&b){return a.cycle<b.cycle;});
   std::vector<std::array<int64_t,6>> outputs;
   std::vector<Event> active_set;
   uint32_t horizon=uint32_t(frames*period+drain), applied=0,seu=0,set=0, excess_outputs=0, pulse_samples=0;
   size_t event_index=0;
   int64_t first_due_cycle=-1;
   auto probe=[&](unsigned domain,unsigned word){top.fault_domain=domain;top.fault_word=word;top.eval();
    std::array<std::uint32_t,6> value{};for(unsigned j=0;j<6;++j)value[j]=top.probe_encoded_out[j];return value;};
   auto write=[&](unsigned domain,unsigned word,const std::array<std::uint32_t,6>&value,unsigned mode){
    top.step=0;top.fault_domain=domain;top.fault_word=word;top.fault_mode=mode;for(unsigned j=0;j<6;++j)top.fault_value[j]=value[j];
    top.fault_valid=1;tick(top);top.fault_valid=0;top.fault_mode=0;};
   auto get_bit=[](const std::array<std::uint32_t,6>&value,unsigned bit){return(value[bit/32]>>(bit%32))&1U;};
   auto put_bit=[](std::array<std::uint32_t,6>&value,unsigned bit,unsigned set){const auto mask=std::uint32_t(1)<<(bit%32);
    value[bit/32]=(value[bit/32]&~mask)|(set?mask:0);};
   for(uint32_t cycle=0;cycle<horizon;cycle++) {
    top.input_valid=cycle%uint32_t(period)==0 && cycle/uint32_t(period)<uint32_t(frames);
    if(top.input_valid) {
     uint32_t frame=cycle/uint32_t(period); top.input_frame=frame;
     top.input0=uint16_t(sample(frame,0,seed)); top.input1=uint16_t(sample(frame,1,seed));
     top.input2=uint16_t(sample(frame,2,seed)); top.input3=uint16_t(sample(frame,3,seed));
    }
    top.set_valid=0; top.set0=0; top.set1=0; top.set2=0; top.set3=0;
    const unsigned first=event_index;std::vector<unsigned> old,pre_values;
    if(joint){
     while(event_index<events.size()&&events[event_index].cycle==cycle)++event_index;
     old.resize(event_index-first);pre_values.resize(event_index-first);std::vector<std::string> missing;
     for(unsigned j=first;j<event_index;++j){const auto&e=events[j];const auto value=probe(e.domain,e.word);
      old[j-first]=get_bit(value,e.bit);captures.preflight(missing,e.response,old[j-first],e.domain,cycle,e.word,e.bit);}
     arh::JointCapture::require_preflight(missing);
     std::map<std::pair<unsigned,unsigned>,std::array<std::uint32_t,6>> pre;
     for(unsigned j=first;j<event_index;++j){const auto&e=events[j];auto key=std::make_pair(e.domain,e.word);
      auto it=pre.find(key);if(it==pre.end())it=pre.emplace(key,probe(e.domain,e.word)).first;
      pre_values[j-first]=captures.pre(e.response,old[j-first],e.domain,cycle,e.word,e.bit);
      put_bit(it->second,e.bit,pre_values[j-first]);}
     for(const auto&v:pre)write(v.first.first,v.first.second,v.second,1);
     applied+=event_index-first;
    }
    while(!joint&&event_index<events.size() && events[event_index].cycle==cycle) {
     auto event=events[event_index++]; applied++;
     if(event.domain==14) {
      if(event.word>=4 || event.bit>=64) throw std::runtime_error("invalid FIR SET node");
      active_set.push_back(event);
      set++;
     } else {
      top.step=0; top.fault_valid=1; top.fault_domain=uint8_t(event.domain);
      top.fault_word=uint16_t(event.word); top.fault_bit=uint8_t(event.bit); top.eval();
      bool before=top.fault_probe; tick(top);
      if(bool(top.fault_probe)==before) throw std::runtime_error("physical FIR XOR did not change its addressed stored bit");
      top.fault_valid=0; seu++;
     }
    }
    active_set.erase(std::remove_if(active_set.begin(),active_set.end(),[&](const Event& event) {
      return cycle-event.cycle>=uint32_t(pulse_cycles);
    }),active_set.end());
    for(const auto& event:active_set) {
     top.set_valid|=uint8_t(1u<<event.word); uint64_t mask=uint64_t(1)<<event.bit;
     if(event.word==0) top.set0^=mask; if(event.word==1) top.set1^=mask;
     if(event.word==2) top.set2^=mask; if(event.word==3) top.set3^=mask;
     pulse_samples++;
    }
    top.step=1; top.eval();
    if(top.output_valid && top.output_ready) {
     if(outputs.size()>=size_t(frames)*2) excess_outputs++;
     else outputs.push_back({int64_t(top.output_frame),int64_t(cycle),int16_t(top.output0),int16_t(top.output1),int16_t(top.output2),int16_t(top.output3)});
    }
    tick(top);
    if(first_due_cycle<0 && top.due_events) first_due_cycle=cycle;
    if(joint){
     // The FI-only helper edges never advance this functional clock. Actual
     // write flags survive helpers and identify the original functional CE.
     const auto span=inputs.at(cycle);
     std::map<std::pair<unsigned,unsigned>,std::array<std::uint32_t,6>> original_d;
     std::map<std::pair<unsigned,unsigned>,std::uint64_t> logical;
     for(auto j=span.first;j<span.second;++j){const auto&e=inputs.event(j);unsigned domain=e.domain/1024,word=e.domain%1024;
      auto value=probe(domain,word);if(e.bit>=top.probe_payload_bits)throw std::runtime_error("FIR original D exceeds actual payload");
      const auto key=std::make_pair(domain,word);const bool ce=top.probe_capture_written;
      const auto data=std::uint64_t(top.probe_data);if(!original_d.count(key))original_d[key]=value;
      auto it=logical.emplace(key,data).first;
      if(inputs.sample(e,(data>>e.bit)&1U,ce))it->second^=std::uint64_t(1)<<e.bit;
     }
     for(const auto&v:logical){probe(v.first.first,v.first.second);if(!top.probe_capture_written)continue;
      std::array<std::uint32_t,6> payload{};payload[0]=std::uint32_t(v.second);payload[1]=std::uint32_t(v.second>>32);
      write(v.first.first,v.first.second,payload,2);const auto actual=probe(v.first.first,v.first.second);
      for(unsigned j=0;j<6;++j)inputs.encoded_updates+=__builtin_popcount(actual[j]^original_d.at(v.first)[j]);}
     std::map<std::pair<unsigned,unsigned>,std::array<std::uint32_t,6>> captured;
     for(unsigned j=first;j<event_index;++j){const auto&e=events[j];auto key=std::make_pair(e.domain,e.word);
      auto it=captured.find(key);if(it==captured.end())it=captured.emplace(key,probe(e.domain,e.word)).first;
      else probe(e.domain,e.word);
      const unsigned d=get_bit(it->second,e.bit),ce=top.probe_capture_written;
      const auto before=old[j-first];const auto result=captures.capture(e.response,before,d,ce,cycle,e.domain,e.word,e.bit,int(pre_values[j-first]));
      put_bit(it->second,e.bit,result);
     }
     for(const auto&v:captured){write(v.first.first,v.first.second,v.second,1);
      if(probe(v.first.first,v.first.second)!=v.second)throw std::runtime_error("FIR actual capture storage differs");}
    }
   }
   int64_t coefficient_sum=0;
   for(int n=0;n<taps;n++) {
    top.fault_domain=4; top.fault_word=uint16_t(n); top.eval();
    uint64_t value=top.probe_data&((uint64_t(1)<<coefficient)-1);
    coefficient_sum+=value>=(uint64_t(1)<<(coefficient-1)) ? int64_t(value)-int64_t(uint64_t(1)<<coefficient) : int64_t(value);
   }
   std::cout<<"{\"seed\":"<<seed<<",\"input_frames\":"<<top.received_frames<<",\"cycles\":"<<horizon
    <<",\"accepted_frames\":"<<top.accepted_frames<<",\"overflow_frames\":"<<top.overflow_frames
    <<",\"recovery_cycles\":"<<top.recovery_cycles<<",\"replayed_frames\":"<<top.replayed_frames
    <<",\"reloaded_coefficients\":"<<top.reloaded_coefficients<<",\"recovery_attempts\":"<<top.recovery_attempts
    <<",\"peak_backlog_frames\":"<<top.peak_backlog_frames<<",\"mac_operations\":"<<top.mac_operations
    <<",\"service_cycles\":"<<top.service_cycles
    <<",\"encoded_write_bits\":"<<top.encoded_write_bits
    <<",\"encoded_writes\":"<<top.encoded_writes<<",\"corrected_reads\":"<<top.corrected_reads
    <<",\"due_events\":"<<top.due_events<<",\"flushed_frames\":"<<top.flushed_frames
    <<",\"pending_frames\":"<<top.pending_frames<<",\"halted\":"<<(top.halted?"true":"false")
    <<",\"physical_xor_checks\":"<<seu<<",\"final_coefficient_sum\":"<<coefficient_sum
    <<",\"applied_events\":"<<applied<<",\"seu_events\":"<<seu<<",\"set_events\":"<<set
    <<",\"set_pulse_samples\":"<<pulse_samples
    <<",\"unrecorded_output_frames\":"<<excess_outputs;
   std::cout<<",\"first_due_cycle\":"<<first_due_cycle;
   if(joint){captures.print(std::cout);inputs.print(std::cout);}
   std::cout<<",\"outputs\":[";
   for(size_t n=0;n<outputs.size();n++) {
    if(n) std::cout<<','; std::cout<<'[';
    for(int j=0;j<6;j++) {if(j)std::cout<<',';std::cout<<outputs[n][j];}
    std::cout<<']';
   }
   std::cout<<"]}"<<std::endl;
   top.final();
  } catch(const std::exception& error) {
   std::cerr<<error.what()<<'\n';
   std::cout<<"{\"error\":"<<std::quoted(error.what())<<"}"<<std::endl;
  }
 }
}
