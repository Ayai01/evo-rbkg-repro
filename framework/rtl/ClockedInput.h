#pragma once
#include <array>
#include <cstdint>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace arh {
struct InputEvent { unsigned cycle,domain,bit,row; };
class ClockedInput {
    std::vector<std::array<int,2>> operators;
    std::vector<InputEvent> events;
    std::vector<std::array<std::int64_t,7>> evidence;
    std::size_t cursor=0;
    static unsigned number(const std::string&s,unsigned maximum) {
        if(s.empty()||s.find_first_not_of("0123456789")!=std::string::npos)throw std::invalid_argument("Unsigned input field required");
        auto v=std::stoull(s);if(v>maximum)throw std::invalid_argument("Clocked input exceeds its finite field");return unsigned(v);
    }
public:
    std::uint64_t checked=0,changed=0,disabled=0,encoded_updates=0;
    void load(const std::vector<std::string>&f,unsigned domains,unsigned maximum_bit=31) {
        if(f.size()<3)throw std::invalid_argument("I requires input operators and events");
        unsigned n=number(f[1],4096),m=number(f[2],1000000);
        if(f.size()!=3+2*n+4*m)throw std::invalid_argument("Incomplete clocked-input record");
        std::vector<std::array<int,2>> next_ops;
        std::vector<InputEvent> next_events;std::set<std::array<unsigned,3>>seen;
        for(unsigned i=0;i<n;++i){std::array<int,2>v;
            for(unsigned j=0;j<2;++j){std::size_t used=0;v[j]=std::stoi(f[3+2*i+j],&used);
                if(used!=f[3+2*i+j].size()||v[j]<-3||v[j]>1)throw std::invalid_argument("Invalid input response");}
            next_ops.push_back(v);
        }
        for(unsigned i=0;i<m;++i){unsigned at=3+2*n+4*i;
            InputEvent e{number(f[at],1000000000),number(f[at+1],domains-1),number(f[at+2],maximum_bit),number(f[at+3],4095)};
            if(e.row>=n||(!next_events.empty()&&e.cycle<next_events.back().cycle))throw std::invalid_argument("Invalid input class or time order");
            if(!seen.insert({e.cycle,e.domain,e.bit}).second)throw std::invalid_argument("Superpose same-input currents first");
            next_events.push_back(e);
        }
        operators.swap(next_ops);events.swap(next_events);reset();
    }
    void reset(){cursor=0;checked=changed=disabled=encoded_updates=0;evidence.clear();}
    std::pair<std::size_t,std::size_t> at(unsigned cycle) {
        auto first=cursor;
        while(cursor<events.size()&&events[cursor].cycle<=cycle){
            if(events[cursor].cycle!=cycle)throw std::runtime_error("Clocked input missed its original edge");++cursor;
        }
        return {first,cursor};
    }
    const InputEvent& event(std::size_t i)const{return events.at(i);}
    bool sample(const InputEvent&e,unsigned data,bool enabled) {
        int response=operators.at(e.row).at(data);
        if(enabled&&response<0)throw std::runtime_error("Unresolved clocked input at actual enabled edge");
        ++checked;disabled+=!enabled;
        bool effect=enabled&&response!=int(data);changed+=effect;
        if(evidence.size()<8)evidence.push_back({e.cycle,e.domain,e.bit,data,unsigned(enabled),response,unsigned(effect)});
        return effect;
    }
    void print(std::ostream&out)const {
        out<<",\"input_checks\":"<<checked<<",\"input_sample_changes\":"<<changed<<",\"input_disabled\":"<<disabled
            <<",\"input_encoded_bit_updates\":"<<encoded_updates<<",\"remaining_input_events\":"<<(events.size()-cursor)<<",\"input_evidence\":[";
        for(unsigned i=0;i<evidence.size();++i){if(i)out<<',';out<<'[';for(unsigned j=0;j<7;++j){if(j)out<<',';out<<evidence[i][j];}out<<']';}
        out<<']';
    }
};
}
