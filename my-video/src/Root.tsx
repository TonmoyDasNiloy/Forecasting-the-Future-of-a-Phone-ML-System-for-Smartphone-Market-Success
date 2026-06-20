import React from "react";
import { Composition } from "remotion";
import { BiAgentIntro } from "./BiAgentIntro";
import { BiAgentDemo } from "./BiAgentDemo";

export const Root: React.FC = () => {
  return (
    <>
      <Composition
        id="BiAgentDemo"
        component={BiAgentDemo}
        durationInFrames={1430}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="BiAgentIntro"
        component={BiAgentIntro}
        durationInFrames={900}
        fps={30}
        width={1920}
        height={1080}
      />
    </>
  );
};
