package capstone.book_grim_ai.controller;

import capstone.book_grim_ai.service.Service;
import lombok.RequiredArgsConstructor;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.io.IOUtils;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.slf4j.Logger;

import java.io.*;
import java.lang.reflect.Array;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.TimeUnit;
import java.util.ArrayList;
import java.util.List;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;

@RestController
@RequiredArgsConstructor
@RequestMapping("/api")
public class Controller {
    private final Service service;
    private final Logger log = LoggerFactory.getLogger(this.getClass().getSimpleName());

    @GetMapping
    public String test() {
        return "this is test";
    }

    @PostMapping(value = "", consumes = { MediaType.APPLICATION_JSON_VALUE,
            MediaType.MULTIPART_FORM_DATA_VALUE }, produces = MediaType.IMAGE_JPEG_VALUE)
    @ResponseStatus(value = HttpStatus.ACCEPTED)
    public @ResponseBody byte[] createCharacter(
            @RequestPart(value = "prompt") String prompt,
            @RequestPart(value = "image") MultipartFile img) throws IOException {
        log.debug("start create character...");
        try {
            log.debug("multipart img : " + img.getBytes());
            log.debug("originalFileName : " + img.getOriginalFilename());
            // ControlNet 돌리기
            File file = new File("/home/origin_img/" + img.getOriginalFilename());
            img.transferTo(file);
            if (file.exists()) {
                log.debug("file exist!");
            }
            log.debug("created image file...");
            File logs = new File("/home/origin_img/log");
            ProcessBuilder pb = new ProcessBuilder("sudo", "python3.10",
                    "/home/ControlNet-with-Anything-v4/book_grim.py", "-img", file.getPath(), "-p", prompt);
            pb.redirectOutput(logs);
            pb.redirectError(logs);
            Process controlnet = pb.start();
            log.debug("start the process...");
            controlnet.waitFor();
            log.debug("end process...");

        } catch (InterruptedException e) {
            log.error(e.getMessage());
            throw new RuntimeException(e);
        }

        byte[] bytes = Files.readAllBytes(Paths.get("/home/ControlNet-with-Anything-v4/results/output.png"));
        log.debug("response... : " + bytes.toString());

        return bytes;
    }

    @PostMapping(value = "/remove", consumes = { MediaType.APPLICATION_JSON_VALUE,
            MediaType.MULTIPART_FORM_DATA_VALUE }, produces = MediaType.IMAGE_JPEG_VALUE)
    @ResponseStatus(value = HttpStatus.ACCEPTED)
    public @ResponseBody byte[] removeBack(
            @RequestPart(value = "character") MultipartFile character) throws IOException {
        String remove_charac_path;
        try {

            String cache_image_path = "/home/g0521sansan/image_processing/cache_img/";

            log.debug("cahe" + cache_image_path);

            log.debug("character img : " + character.getBytes());
            log.debug("character originalFileName : " + character.getOriginalFilename());

            File charac_file = new File(cache_image_path + character.getOriginalFilename());

            character.transferTo(charac_file);

            File logs = new File(cache_image_path + "log");

            log.debug("remove character back_ground...");
            ProcessBuilder rm = new ProcessBuilder("/usr/bin/python3", "/home/g0521sansan/image_processing/remove.py",
                    charac_file.getPath());
            log.debug("check command : " + rm.command());
            log.debug("charac_file path : " + charac_file.getPath());
            rm.redirectOutput(logs);
            rm.redirectError(logs);
            Process remove = rm.start();
            log.debug("start remove...");
            remove.waitFor();
            log.debug("end remove...");

            remove_charac_path = cache_image_path + FilenameUtils.removeExtension(character.getOriginalFilename())
                    + "_rm." + FilenameUtils.getExtension(character.getOriginalFilename());
            log.debug("cahe imge  path : " + cache_image_path);
            log.debug("charac name " + FilenameUtils.removeExtension(character.getOriginalFilename()));
            log.debug("remomve charac path :" + remove_charac_path);

        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        byte[] bytes = Files.readAllBytes(Paths.get(remove_charac_path));
        log.debug("response... : " + bytes.toString());

        return bytes;
    }

    @PostMapping(value = "/createPage", consumes = { MediaType.APPLICATION_JSON_VALUE,
            MediaType.MULTIPART_FORM_DATA_VALUE }, produces = MediaType.IMAGE_JPEG_VALUE)
    @ResponseStatus(value = HttpStatus.ACCEPTED)
    public @ResponseBody byte[] createPage(
            @RequestPart(value = "back") MultipartFile back,
            @RequestPart(value = "character") MultipartFile character,
            @RequestPart(value = "x") int x,
            @RequestPart(value = "y") int y,
            @RequestPart(value = "feature") String feature) throws IOException {
        log.debug("start Page character...");
        try {

            // 어차피 이미 모델에 캐릭터 학습 돼있는 상태라 제거해도 될듯
            String cache_image_path = "/home/g0521sansan/image_processing/cache_img/";

            log.info("cahe" + cache_image_path);
            log.info("back img : " + back.getBytes());
            log.info("back originalFileName : " + back.getOriginalFilename());

            log.info("character img : " + character.getBytes());
            log.info("character originalFileName : " + character.getOriginalFilename());

            // create Character Variation test

            String charac_image_path = characterVariation(
                    FilenameUtils.removeExtension(character.getOriginalFilename()),
                    feature);
            log.info(charac_image_path);

            // cache_image_path -> chara_image_path 일단은 함수 구현 후 테스트하고 교체

            File back_file = new File(cache_image_path + back.getOriginalFilename());
            File charac_file = new File(cache_image_path + character.getOriginalFilename());

            back.transferTo(back_file);
            character.transferTo(charac_file);

            log.info("created image file...");
            log.info("end create image");

            File mlogs = new File(cache_image_path + "mlog");

            log.info("merge image...");
            ProcessBuilder mg = new ProcessBuilder("python3", "/home/g0521sansan/image_processing/merge.py",
                    back_file.getPath(), charac_image_path, Integer.toString(x), Integer.toString(y));
            mg.redirectOutput(mlogs);
            mg.redirectError(mlogs);
            Process merge = mg.start();
            log.info("start merge...");
            merge.waitFor();
            log.info("end merge...");

        } catch (InterruptedException e) {
            log.error(e.getMessage());

        }
        byte[] bytes = Files.readAllBytes(Paths.get("/home/g0521sansan/image_processing/story.png"));
        log.info("response... : " + bytes.toString());

        return bytes;

    }

    @PostMapping(value = "/checkGrammar", consumes = MediaType.MULTIPART_FORM_DATA_VALUE, produces = MediaType.APPLICATION_JSON_VALUE)
    @ResponseStatus(value = HttpStatus.ACCEPTED)
    public @ResponseBody ArrayList<String> checkGrammar(
            @RequestPart(value = "content") String content) throws IOException, InterruptedException {
        ProcessBuilder check = new ProcessBuilder("python3", "/home/g0521sansan/py-hanspell/korean_check.py", "--text",
                content);
        Process kocheck = check.start();
        kocheck.waitFor();
        ArrayList<String> spellList = new ArrayList<String>();

        // File file = new File("E:\\2023.1\\캡스톤\\py-hanspell\\spellList.txt");
        File file = new File("/home/g0521sansan/py-hanspell/spellList.txt");
        if (file.exists()) {
            BufferedReader inFile = new BufferedReader(new InputStreamReader(new FileInputStream(file), "utf-8"));
            String sLine = "";

            while ((sLine = inFile.readLine()) != null) {
                spellList.add(sLine);
            }
            if (spellList.isEmpty()) {
                System.out.println("Empty!");
            } else {
                for (String o : spellList) {
                    System.out.println(o);
                }
            }
        }
        return spellList;
    }

    public String characterVariation(String character, String features) throws IOException, InterruptedException {

        String resultPath = "/home/super/Desktop/stable-diffusion/result.png";

        log.info("\""+character+", "+features.substring(1,features.length())+"\"");
        ProcessBuilder variation = new ProcessBuilder(
                "python3",
                "/home/super/Desktop/stable-diffusion/scripts/txt2img.py",
                "--prompt",
                "\""+character+", "+features.substring(1,features.length())+"\"",
                "--H",
                "512",
                "--W",
                "512",
                "--outdir",
                "./",
                "--n_samples",
                "1",
                "--ckpt",
                "/home/super/Desktop/dreambooth/content/MyDrive/Fast-Dreambooth/Sessions/character/character.ckpt",
                "--config",
                "/home/super/Desktop/stable-diffusion/configs/stable-diffusion/v1-inference.yaml"

        );

        File logs = new File("/home/super/Desktop/stable-diffusion/result.png"+ "log");
        variation.redirectOutput(logs);
        variation.redirectError(logs);


        variation.redirectOutput();
        Process p = variation.start();
        log.info("start variation...");
        p.waitFor();
        log.info("end variation...");



        ProcessBuilder rm = new ProcessBuilder("/usr/bin/python3", "/home/g0521sansan/image_processing/remove.py",
                resultPath);

        Process remove = rm.start();
        remove.waitFor();

        resultPath = "/home/g0521sansan/image_processing/cache_img/result_rm.png";
        log.info("After variation remove : "+resultPath);
        return resultPath;
    }
    
}
