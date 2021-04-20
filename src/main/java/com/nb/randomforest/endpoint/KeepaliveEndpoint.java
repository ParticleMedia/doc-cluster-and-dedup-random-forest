package com.nb.randomforest.endpoint;

import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseBody;
import org.springframework.stereotype.Controller;

@Controller
@ResponseBody
public class KeepaliveEndpoint {

    @RequestMapping(path = "/keepalive")
	public String keepalive(){
        return "OK";
    }
}